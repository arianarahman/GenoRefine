# Purpose: Plan, launch, and resume the pinned multi-section GenoRefine and SpaGCN spatial panel.
# Author: Ariana Rahman (Arizona State University)

param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$')][string]$Prefix,
    [string]$Distro = $env:GENOREFINE_WSL_DISTRO,
    [string]$EvaluationPython = $env:GENOREFINE_EVALUATION_PYTHON,
    [string]$HarmonyPython = $env:GENOREFINE_HARMONY_PYTHON,
    [string]$GenoRefineImage = 'genorefine-gpu:tf25.02',
    [string]$SpaGCNImage = 'genorefine-spagcn:1.2.7-panel-v1',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
# Resolve the frozen runtime contract and the six-section execution plan.
if (-not $Distro) { $Distro = 'Ubuntu-24.04' }
if (-not $EvaluationPython) { $EvaluationPython = 'python3' }
if (-not $HarmonyPython) { $HarmonyPython = 'python' }
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Runs = Join-Path $Project 'revision_pipeline\runs'
$LinuxProject = (& wsl.exe -d $Distro -- wslpath -a $Project).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProject) { throw 'Could not resolve the project path in WSL.' }
$HarmonyId = "$Prefix-harmony-fixed10"
$KId = "$Prefix-k-selection"
$Harmony = "revision_pipeline/runs/$HarmonyId"
$KSelection = "revision_pipeline/runs/$KId"
$Sections = @('151507', '151508', '151669', '151670', '151673', '151674')
$Donors = @('Br5292', 'Br5595', 'Br8100')
$Spec = Get-Content -LiteralPath (Join-Path $Project 'revision_pipeline\configs\spatial_multisection_panel_v1.json') -Raw | ConvertFrom-Json
$ExpectedSpaGCNImage = $Spec.spagcn.runtime.image_tag
$ExpectedSpaGCNImageId = $Spec.spagcn.runtime.image_id
$SpaGCNImageId = $null
$ExpectedGenoRefineImage = $Spec.genorefine.runtime.image_tag
$ExpectedGenoRefineImageId = $Spec.genorefine.runtime.image_id
$GenoRefineImageId = $null

if ($SpaGCNImage -ne $ExpectedSpaGCNImage) {
    throw "SpaGCN image tag must match the frozen protocol: $ExpectedSpaGCNImage"
}
if ($GenoRefineImage -ne $ExpectedGenoRefineImage) {
    throw "GenoRefine image tag must match the frozen protocol: $ExpectedGenoRefineImage"
}
if ($Distro -ne $Spec.evaluation.runtime.wsl_distribution) {
    throw 'Evaluation WSL distribution must match the frozen protocol'
}

# Plan mode reports the complete run graph without creating scientific outputs.
if ($PlanOnly) {
    [pscustomobject]@{
        Prefix = $Prefix
        FixedHarmonyJobs = 1
        LabelFreeKJobs = 1
        GenoRefineTrainingJobs = 5
        SpaGCNTrainingJobs = 30
        SectionScoringJobs = 72
        DonorMixingScoringJobs = 21
        ConsolidationJobs = 1
        TotalRunDirectories = 131
        EvaluationEnvironment = "$Distro $EvaluationPython"
        HarmonyEnvironment = $HarmonyPython
        GenoRefineEnvironment = $GenoRefineImage
        GenoRefineExpectedImageId = $ExpectedGenoRefineImageId
        SpaGCNEnvironment = $SpaGCNImage
        SpaGCNExpectedImageId = $ExpectedSpaGCNImageId
    } | Format-List
    return
}

# A prefix-scoped lock prevents concurrent writers from sharing an output namespace.
$LockPath = Join-Path $Runs ".orchestrator-$Prefix.lock"
try {
    $LockStream = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate,
                                  [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Another Package 4 orchestrator holds the prefix lock: $LockPath"
}

$LocationPushed = $false
try {
Push-Location -LiteralPath $Project
$LocationPushed = $true

# Resume helpers preserve stale attempts and deeply verify every reusable completed run.
function Move-StaleIncompleteRun([string]$Id) {
    $candidate = Join-Path $Runs ".incomplete-$Id"
    if (-not (Test-Path -LiteralPath $candidate)) { return }
    $resolvedCandidate = (Resolve-Path -LiteralPath $candidate).Path
    $resolvedRuns = (Resolve-Path -LiteralPath $Runs).Path
    if ([IO.Path]::GetDirectoryName($resolvedCandidate) -ne $resolvedRuns) {
        throw "Refusing to move incomplete run outside the locked run directory: $resolvedCandidate"
    }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $suffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $target = Join-Path $resolvedRuns ".retired-$Id-$stamp-$suffix"
    if ([IO.Path]::GetDirectoryName($target) -ne $resolvedRuns -or (Test-Path -LiteralPath $target)) {
        throw "Invalid stale-run archive target: $target"
    }
    Move-Item -LiteralPath $resolvedCandidate -Destination $target
    Write-Host "Preserved stale incomplete run as $target"
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
    & $HarmonyPython -m revision_pipeline.spatial_multisection.verify_completed `
        --run "revision_pipeline/runs/$Id" --kind $Kind
    if ($LASTEXITCODE -ne 0) { throw "Completed run failed deep resume validation: $Id" }
    return $true
}

function Invoke-Evaluation([string[]]$Arguments) {
    $command = "cd '$LinuxProject' && '$EvaluationPython' -m " + ($Arguments -join ' ')
    & wsl.exe -d $Distro -- bash -lc $command
    if ($LASTEXITCODE -ne 0) { throw "Evaluation command failed: $command" }
}

function Invoke-DockerPython([string]$Image, [string[]]$Arguments) {
    $dockerArgs = @('run', '--rm', '--gpus', 'all', '--shm-size', '12g',
                    '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                    '-v', "${Project}:/workspace", '-w', '/workspace')
    if ($Image -eq $SpaGCNImage) {
        if (-not $SpaGCNImageId) { throw 'SpaGCN image identity was not validated' }
        $dockerArgs += @('-e', "SPAGCN_IMAGE_ID=$SpaGCNImageId")
    } elseif ($Image -eq $GenoRefineImage) {
        if (-not $GenoRefineImageId) { throw 'GenoRefine image identity was not validated' }
        $dockerArgs += @('-e', "GENOREFINE_IMAGE_ID=$GenoRefineImageId")
    }
    $dockerArgs += @($Image, 'python', '-m') + $Arguments
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed in $Image" }
}

# Validate both pinned GPU images and the evaluation interpreter before execution.
if (-not (Get-Command $HarmonyPython -ErrorAction SilentlyContinue)) {
    throw "Harmony interpreter was not found: $HarmonyPython"
}
& docker image inspect $GenoRefineImage *> $null
if ($LASTEXITCODE -ne 0) { throw "Frozen GenoRefine image is missing: $GenoRefineImage" }
$GenoRefineImageId = (& docker image inspect $GenoRefineImage --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0 -or $GenoRefineImageId -ne $ExpectedGenoRefineImageId) {
    throw "GenoRefine image ID mismatch: expected $ExpectedGenoRefineImageId; observed $GenoRefineImageId"
}

# Build the shared Harmony foundation, choose label-free K, and train both candidate methods.
if (-not (Test-Completed $HarmonyId 'spatial_multisection_harmony_fixed')) {
    & $HarmonyPython -m revision_pipeline.spatial_multisection.harmony_fixed --run-id $HarmonyId --execute
    if ($LASTEXITCODE -ne 0) { throw 'Fixed-10 Harmony failed' }
}

if (-not (Test-Completed $KId 'spatial_multisection_k_selection')) {
    Invoke-Evaluation @('revision_pipeline.spatial_multisection.k_selection', '--harmony', $Harmony,
                        '--run-id', $KId, '--execute')
}

foreach ($seed in 0..4) {
    $id = "$Prefix-gr-s$seed"
    if (-not (Test-Completed $id 'spatial_multisection_genorefine_training')) {
        Invoke-DockerPython $GenoRefineImage @('revision_pipeline.spatial_multisection.train_genorefine',
            '--harmony', $Harmony, '--k-selection', $KSelection, '--seed', "$seed",
            '--run-id', $id, '--runtime-profile', 'fast_gpu', '--execute')
    }
}

& docker image inspect $SpaGCNImage *> $null
if ($LASTEXITCODE -ne 0) {
    & docker build -f (Join-Path $PSScriptRoot 'Dockerfile.spagcn-gpu') -t $SpaGCNImage $Project
    if ($LASTEXITCODE -ne 0) { throw 'SpaGCN GPU image build failed' }
}
$SpaGCNImageId = (& docker image inspect $SpaGCNImage --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0 -or $SpaGCNImageId -ne $ExpectedSpaGCNImageId) {
    throw "SpaGCN image ID mismatch: expected $ExpectedSpaGCNImageId; observed $SpaGCNImageId"
}
Invoke-DockerPython $SpaGCNImage @('revision_pipeline.spatial_multisection.spagcn_runtime')

# Train one histology-aware SpaGCN model for every section and algorithmic seed.
foreach ($section in $Sections) {
    foreach ($seed in 0..4) {
        $id = "$Prefix-spagcn-$section-s$seed"
        if (-not (Test-Completed $id 'spatial_multisection_spagcn_training')) {
            Invoke-DockerPython $SpaGCNImage @('revision_pipeline.spatial_multisection.train_spagcn',
                '--section', $section, '--seed', "$seed", '--harmony', $Harmony,
                '--k-selection', $KSelection, '--run-id', $id, '--device', 'auto', '--execute')
        }
    }
}

# Score section-level representation and task-native endpoints.
foreach ($section in $Sections) {
    foreach ($pair in @(@('harmony_fixed', 'harmony-fixed'), @('harmony_native_sensitivity', 'harmony-native'))) {
        $id = "$Prefix-score-section-$section-$($pair[1])"
        if (-not (Test-Completed $id 'spatial_multisection_section_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_multisection.score', 'section', '--method', $pair[0],
                '--section', $section, '--harmony', $Harmony, '--k-selection', $KSelection,
                '--run-id', $id, '--execute')
        }
    }
    foreach ($seed in 0..4) {
        $grId = "$Prefix-score-section-$section-gr-s$seed"
        if (-not (Test-Completed $grId 'spatial_multisection_section_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_multisection.score', 'section', '--method', 'genorefine',
                '--section', $section, '--harmony', $Harmony, '--k-selection', $KSelection,
                '--training', "revision_pipeline/runs/$Prefix-gr-s$seed", '--run-id', $grId, '--execute')
        }
        $spaId = "$Prefix-score-section-$section-spagcn-s$seed"
        if (-not (Test-Completed $spaId 'spatial_multisection_section_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_multisection.score', 'section', '--method', 'spagcn',
                '--section', $section, '--harmony', $Harmony, '--k-selection', $KSelection,
                '--training', "revision_pipeline/runs/$Prefix-spagcn-$section-s$seed", '--run-id', $spaId, '--execute')
        }
    }
}

# Aggregate cross-section mixing within each donor before final consolidation.
foreach ($donor in $Donors) {
    foreach ($pair in @(@('harmony_fixed', 'harmony-fixed'), @('harmony_native_sensitivity', 'harmony-native'))) {
        $id = "$Prefix-score-donor-$donor-$($pair[1])"
        if (-not (Test-Completed $id 'spatial_multisection_donor_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_multisection.score', 'donor', '--method', $pair[0],
                '--donor', $donor, '--harmony', $Harmony, '--k-selection', $KSelection,
                '--run-id', $id, '--execute')
        }
    }
    foreach ($seed in 0..4) {
        $id = "$Prefix-score-donor-$donor-gr-s$seed"
        if (-not (Test-Completed $id 'spatial_multisection_donor_score')) {
            Invoke-Evaluation @('revision_pipeline.spatial_multisection.score', 'donor', '--method', 'genorefine',
                '--donor', $donor, '--harmony', $Harmony, '--k-selection', $KSelection,
                '--training', "revision_pipeline/runs/$Prefix-gr-s$seed", '--run-id', $id, '--execute')
        }
    }
}

$PanelId = "$Prefix-full-panel"
if (-not (Test-Completed $PanelId 'spatial_multisection_panel')) {
    Invoke-Evaluation @('revision_pipeline.spatial_multisection.consolidate', '--prefix', $Prefix,
                        '--run-id', $PanelId, '--execute')
}

Write-Output (Join-Path $Runs $PanelId)
} finally {
    if ($LocationPushed) { Pop-Location }
    $LockStream.Dispose()
}
