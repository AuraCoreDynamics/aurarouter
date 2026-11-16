$ErrorActionPreference = "Stop"

$FoundPython = $null

function Test-ValidPython($ExePath) {
    if (-not $ExePath -or -not (Test-Path $ExePath)) { return $false }
    try {
        $item = Get-Item $ExePath -ErrorAction SilentlyContinue
        if ($item.Length -eq 0) { return $false }
        $res = & $ExePath -c "import sys; print(sys.version_info[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $res -match "^\d+") {
            return $true
        }
    } catch {}
    return $false
}

# 1. Try PATH
$PythonInPath = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($PythonInPath -and (Test-ValidPython $PythonInPath.Source)) {
    $FoundPython = $PythonInPath.Source
}

# 2. Try Registry PEP 514
if (-not $FoundPython) {
    foreach ($RegPath in @("HKCU:\Software\Python\PythonCore", "HKLM:\Software\Python\PythonCore")) {
        if (Test-Path $RegPath) {
            foreach ($Version in (Get-ChildItem -Path $RegPath -ErrorAction SilentlyContinue)) {
                $InstallPathKey = Join-Path $Version.PSPath "InstallPath"
                if (Test-Path $InstallPathKey) {
                    $InstallPath = (Get-ItemProperty -Path $InstallPathKey).'(default)'
                    $ExePath = Join-Path $InstallPath "python.exe"
                    if (Test-ValidPython $ExePath) {
                        $FoundPython = $ExePath
                        break
                    }
                }
            }
        }
        if ($FoundPython) { break }
    }
}

# 3. Check sibling auraseed bootstrapped Python
if (-not $FoundPython) {
    $CheckRepoExe = Join-Path $PSScriptRoot "..\auraseed\.bootstrapped_python\python.exe"
    if (Test-ValidPython $CheckRepoExe) {
        $FoundPython = $CheckRepoExe
    }
}

if (-not $FoundPython) {
    Write-Error "[AuraRouter Bootstrapper] Error: Could not locate a valid base Python executable on the host."
    exit 1
}

$SrcDir = Join-Path $PSScriptRoot "src"
$PythonPaths = @($SrcDir)

$ParentDir = Split-Path $PSScriptRoot -Parent
foreach ($Dep in @("auragrid-py", "auragrid-api", "benchwarmer")) {
    $DepDir = Join-Path $ParentDir $Dep
    if (Test-Path $DepDir) {
        $DepSrc = Join-Path $DepDir "src"
        $PythonPaths += $(if (Test-Path $DepSrc) { $DepSrc } else { $DepDir })
    }
}

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = (($PythonPaths -join ";") + ";$env:PYTHONPATH")
} else {
    $env:PYTHONPATH = ($PythonPaths -join ";")
}

& $FoundPython -m aurarouter $args
exit $LASTEXITCODE

