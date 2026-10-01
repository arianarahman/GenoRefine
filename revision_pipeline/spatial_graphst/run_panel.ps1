# Purpose: Plan, launch, and resume the pinned GraphST donor-pair spatial panel.
# Author: Ariana Rahman (Arizona State University)

param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$')][string]$Prefix,
    [string]$Distro = $env:GENOREFINE_WSL_DISTRO,
    [string]$EvaluationPython = $env:GENOREFINE_EVALUATION_PYTHON,
    [string]$GraphSTImage = 'genorefine-graphst:1.1.1-panel-v1',
    [switch]$PlanOnly,
    [switch]$PreflightOnly,
    [switch]$ExecutePanel
)

$ErrorActionPreference = 'Stop'
# Resolve the frozen runtime contract and the donor/section execution plan.
if (-not $Distro) { $Distro = 'Ubuntu-24.04' }
if (-not $EvaluationPython) { $EvaluationPython = 'python3' }
if ((@($PlanOnly, $PreflightOnly, $ExecutePanel) | Where-Object { $_ }).Count -ne 1) {
    throw 'Choose exactly one of -PlanOnly, -PreflightOnly, or -ExecutePanel'
}
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Runs = Join-Path $Project 'revision_pipeline\runs'
$LinuxProject = (& wsl.exe -d $Distro -- wslpath -a $Project).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProject) { throw 'Could not resolve the project path in WSL.' }
$SpecPath = Join-Path $Project 'revision_pipeline\configs\spatial_graphst_panel_v1.json'
$Spec = Get-Content -LiteralPath $SpecPath -Raw | ConvertFrom-Json
$ExpectedImage = $Spec.runtime.image_tag
$ExpectedImageId = $Spec.runtime.image_id
$Sections = @('151507', '151508', '151669', '151670', '151673', '151674')
$Donors = @('Br5292', 'Br5595', 'Br8100')
$DonorSections = @{
    'Br5292' = @('151507', '151508')
    'Br5595' = @('151669', '151670')
    'Br8100' = @('151673', '151674')
}
$KId = "$Prefix-k-selection"
$KSelection = "revision_pipeline/runs/$KId"
$PreflightId = "$Prefix-preflight"

if ($GraphSTImage -ne $ExpectedImage) {
    throw "GraphST image tag must match the frozen protocol: $ExpectedImage"
}
if ($Distro -ne $Spec.evaluation.runtime.wsl_distribution) {
    throw 'Evaluation WSL distribution must match the frozen protocol'
}

# Plan mode exposes the complete job count without creating scientific runs.
if ($PlanOnly) {
    [pscustomobject]@{
        Prefix = $Prefix
        PASTEAlignmentJobs = 3
        LabelFreeKJobs = 1
        GraphSTTrainingJobs = 15
        SectionScoringJobs = 30
        DonorMixingScoringJobs = 15
        ConsolidationJobs = 1
        ScientificRunDirectories = 65
        SeparatePreflightRunDirectories = 1
        GraphSTSource = "1.1.1 @ $($Spec.sources.graphst.commit)"
        PASTESource = "1.4.0 @ $($Spec.sources.paste.commit)"
        GraphSTImage = $ExpectedImage
        ExpectedImageId = $ExpectedImageId
        EvaluationEnvironment = "$Distro $EvaluationPython"
        FullExecutionAuthorizedByThisInvocation = $false
    } | Format-List
    return
}

# A prefix-scoped lock and transcript make interrupted executions safely resumable.
$LockPath = Join-Path $Runs ".orchestrator-$Prefix-graphst4b.lock"
try {
    $LockStream = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate,
                                  [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Another GraphST Package 4b orchestrator holds the prefix lock: $LockPath"
}

$LocationPushed = $false
$TranscriptStarted = $false
$Orchestration = $null
$CurrentStage = 'initialization'
try {
Push-Location -LiteralPath $Project
$LocationPushed = $true
$Orchestration = Join-Path $Runs "_orchestration-$Prefix-graphst4b"
if (-not (Test-Path -LiteralPath $Orchestration)) {
    New-Item -ItemType Directory -Path $Orchestration | Out-Null
}
$TranscriptPath = Join-Path $Orchestration 'console-transcript.log'
Start-Transcript -LiteralPath $TranscriptPath -Append | Out-Null
$TranscriptStarted = $true

# Persist machine-readable progress and retire only verified stale incomplete runs.
function Write-State([string]$Stage, [string]$Status) {
    $script:CurrentStage = $Stage
    [pscustomobject]@{
        schema_version = 1
        prefix = $Prefix
        stage = $Stage
        status = $Status
        updated_at_utc = [DateTime]::UtcNow.ToString('o')
        scientific_panel_requested = [bool]$ExecutePanel
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Orchestration 'state.json') -Encoding utf8
}

function Move-StaleIncompleteRun([string]$Id) {
    $candidate = Join-Path $Runs ".incomplete-$Id"
    if (-not (Test-Path -LiteralPath $candidate)) { return }
    $activeContainers = @(& docker ps --filter "label=genorefine.graphst4b.run_id=$Id" --format '{{.ID}}') |
        Where-Object { $_ -and $_.Trim() }
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect active GraphST containers' }
    if ($activeContainers.Count -gt 0) {
        throw "Refusing to retire $candidate while its GraphST container is active"
    }
    $pythonProcesses = @(& wsl.exe -d $Distro -- ps -eo args=) |
        Where-Object { $_ -match 'python' -and $_ -match [regex]::Escape($Id) }
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect WSL evaluation processes' }
    if ($pythonProcesses.Count -gt 0) {
        throw "Refusing to retire $candidate while its evaluation process is active"
    }
    $resolvedCandidate = (Resolve-Path -LiteralPath $candidate).Path
    $resolvedRuns = (Resolve-Path -LiteralPath $Runs).Path
    if ([IO.Path]::GetDirectoryName($resolvedCandidate) -ne $resolvedRuns) {
        throw "Refusing to move incomplete run outside the run directory: $resolvedCandidate"
    }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $target = Join-Path $resolvedRuns ".retired-$Id-$stamp-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
    if ([IO.Path]::GetDirectoryName($target) -ne $resolvedRuns -or (Test-Path -LiteralPath $target)) {
        throw "Invalid stale-run archive target: $target"
    }
    Move-Item -LiteralPath $resolvedCandidate -Destination $target
}

function Invoke-Evaluation([string[]]$Arguments) {
    $command = "cd '$LinuxProject' && '$EvaluationPython' -m " + ($Arguments -join ' ')
    & wsl.exe -d $Distro -- bash -lc $command
    if ($LASTEXITCODE -ne 0) { throw "Evaluation command failed: $command" }
}

function Invoke-GraphST([string[]]$Arguments, [bool]$UseGpu) {
    $dockerArgs = @('run', '--rm', '--shm-size', '16g',
                    '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                    '-e', "GRAPHST_IMAGE_ID=$ObservedImageId",
                    '-v', "${Project}:/workspace", '-w', '/workspace')
    $runIdIndex = [Array]::IndexOf($Arguments, '--run-id')
    if ($runIdIndex -ge 0) {
        if ($runIdIndex + 1 -ge $Arguments.Count) { throw 'Missing value after --run-id' }
        $containerRunId = $Arguments[$runIdIndex + 1]
        $dockerArgs += @('--label', "genorefine.graphst4b.run_id=$containerRunId")
    }
    if ($UseGpu) { $dockerArgs += @('--gpus', 'all') }
    $dockerArgs += @($GraphSTImage, 'python', '-m') + $Arguments
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "GraphST container command failed: $($Arguments -join ' ')" }
}

function Test-Completed([string]$Id, [string]$Kind) {
    $manifest = Join-Path $Runs "$Id\run.json"
    if (-not (Test-Path -LiteralPath $manifest)) {
        Move-StaleIncompleteRun $Id
        return $false
    }
    $record = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    if ($record.status -ne 'succeeded' -or $record.kind -ne $Kind) {
        throw "Existing run $Id is not a completed $Kind run"
    }
    Invoke-Evaluation @('revision_pipeline.spatial_graphst.verify_completed',
                        '--run', "revision_pipeline/runs/$Id", '--kind', $Kind)
    return $true
}

# Verify source archives, the pinned container identity, and CUDA before training.
Write-State 'runtime_validation' 'running'
& python -m revision_pipeline.spatial_graphst.source_acquisition --all --execute
if ($LASTEXITCODE -ne 0) { throw 'Official-source lock verification failed' }
& docker image inspect $GraphSTImage *> $null
if ($LASTEXITCODE -ne 0) { throw "Frozen GraphST image is missing: $GraphSTImage" }
$ObservedImageId = (& docker image inspect $GraphSTImage --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0 -or $ObservedImageId -ne $ExpectedImageId) {
    throw "GraphST image ID mismatch: expected $ExpectedImageId; observed $ObservedImageId"
}
Invoke-GraphST @('revision_pipeline.spatial_graphst.runtime', '--require-cuda') $true
Write-State 'runtime_validation' 'succeeded'

if (-not (Test-Completed $PreflightId 'spatial_graphst_preflight')) {
    Write-State 'preflight' 'running'
    Invoke-GraphST @('revision_pipeline.spatial_graphst.preflight',
                     '--run-id', $PreflightId, '--execute') $true
}
if (-not (Test-Completed $PreflightId 'spatial_graphst_preflight')) {
    throw 'GraphST preflight did not publish a valid completed run'
}
Write-State 'preflight' 'succeeded'
if ($PreflightOnly) {
    Write-Output (Join-Path $Runs $PreflightId)
    return
}

# Execute the scientific stages in dependency order; each stage reuses only deeply verified runs.
Write-State 'alignments' 'running'
foreach ($donor in $Donors) {
    $id = "$Prefix-align-$donor"
    if (-not (Test-Completed $id 'spatial_graphst_alignment')) {
        Invoke-GraphST @('revision_pipeline.spatial_graphst.align', '--donor', $donor,
                         '--run-id', $id, '--execute') $false
    }
}
Write-State 'alignments' 'succeeded'

Write-State 'label_free_k_selection' 'running'
if (-not (Test-Completed $KId 'spatial_graphst_k_selection')) {
    Invoke-Evaluation @('revision_pipeline.spatial_graphst.k_selection',
                        '--run-id', $KId, '--execute')
}
Write-State 'label_free_k_selection' 'succeeded'

Write-State 'training' 'running'
foreach ($donor in $Donors) {
    foreach ($seed in 0..4) {
        $id = "$Prefix-graphst-$donor-s$seed"
        if (-not (Test-Completed $id 'spatial_graphst_training')) {
            Invoke-GraphST @('revision_pipeline.spatial_graphst.train',
                '--donor', $donor, '--seed', "$seed",
                '--alignment', "revision_pipeline/runs/$Prefix-align-$donor",
                '--k-selection', $KSelection, '--run-id', $id, '--device', 'auto', '--execute') $true
        }
    }
}
Write-State 'training' 'succeeded'

Write-State 'scoring' 'running'
foreach ($donor in $Donors) {
    foreach ($seed in 0..4) {
        $training = "revision_pipeline/runs/$Prefix-graphst-$donor-s$seed"
        $alignment = "revision_pipeline/runs/$Prefix-align-$donor"
        foreach ($section in $DonorSections[$donor]) {
            $id = "$Prefix-score-section-$section-s$seed"
            if (-not (Test-Completed $id 'spatial_graphst_section_score')) {
                Invoke-Evaluation @('revision_pipeline.spatial_graphst.score', 'section',
                    '--section', $section, '--donor', $donor, '--seed', "$seed",
                    '--training', $training, '--alignment', $alignment,
                    '--k-selection', $KSelection, '--run-id', $id, '--execute')
            }
        }
        $donorId = "$Prefix-score-donor-$donor-s$seed"
        if (-not (Test-Completed $donorId 'spatial_graphst_donor_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_graphst.score', 'donor',
                '--donor', $donor, '--seed', "$seed", '--training', $training,
                '--alignment', $alignment, '--k-selection', $KSelection,
                '--run-id', $donorId, '--execute')
        }
    }
}
Write-State 'scoring' 'succeeded'

$PanelId = "$Prefix-full-panel"
Write-State 'consolidation' 'running'
if (-not (Test-Completed $PanelId 'spatial_graphst_panel')) {
    Invoke-Evaluation @('revision_pipeline.spatial_graphst.consolidate',
                        '--prefix', $Prefix, '--run-id', $PanelId, '--execute')
}
Write-State 'complete' 'succeeded'
Write-Output (Join-Path $Runs $PanelId)
} catch {
    if ($Orchestration -and (Test-Path -LiteralPath $Orchestration)) {
        [pscustomobject]@{
            schema_version = 1
            prefix = $Prefix
            stage = $CurrentStage
            status = 'failed'
            failed_at_utc = [DateTime]::UtcNow.ToString('o')
            scientific_panel_requested = [bool]$ExecutePanel
            exception = $_.Exception.ToString()
            script_stack_trace = $_.ScriptStackTrace
            transcript = 'console-transcript.log'
        } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Orchestration 'failure.json') -Encoding utf8
        Write-State $CurrentStage 'failed'
    }
    throw
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    if ($LocationPushed) { Pop-Location }
    $LockStream.Dispose()
}
