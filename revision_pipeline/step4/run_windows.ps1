param(
    [Parameter(Mandatory=$true)][string]$Acceptance,
    [string]$Distro = $env:GENOREFINE_WSL_DISTRO,
    [string]$Python = $env:GENOREFINE_PRIMARY_PYTHON
)
$ErrorActionPreference = 'Stop'
if (-not $Distro) { $Distro = 'Ubuntu-24.04' }
if (-not $Python) { $Python = 'python3' }
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LinuxProject = (& wsl.exe -d $Distro -- wslpath -a $Project).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProject) { throw 'Could not resolve the project path in WSL.' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class GenoRefineStep4Awake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$awakeRequest = [GenoRefineStep4Awake]::SetThreadExecutionState([uint32]2147483649)
if ($awakeRequest -eq 0) { throw 'Could not establish scoped automatic-sleep prevention.' }
try {
    & wsl.exe -d $Distro --cd $LinuxProject -- env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 JAX_PLATFORMS=cpu $Python -B -X faulthandler -m revision_pipeline.step4.run --execute-step4a --acceptance $Acceptance
    $panelExit = $LASTEXITCODE
    if ($panelExit -ne 0) { throw "Step 4A stopped with code $panelExit. See retained logs." }
}
finally {
    $releasedRequest = [GenoRefineStep4Awake]::SetThreadExecutionState([uint32]2147483648)
}
