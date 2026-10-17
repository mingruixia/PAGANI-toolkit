# Assemble the portable Windows package from a CMake build directory.
param(
    [string]$BuildDir = "build",
    [string]$OutDir = "dist"
)
$ErrorActionPreference = "Stop"
$version = (Select-String -Path "CMakeLists.txt" -Pattern 'project\(PAGANI_Toolkit VERSION ([0-9.]+)').Matches[0].Groups[1].Value
$name = "PAGANI-Toolkit-$version-win64"
$stage = Join-Path $OutDir $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Copy-Item -Recurse (Join-Path $BuildDir "package\*") $stage
$gui = Join-Path $stage "PAGANI.exe"
if (Test-Path $gui) {
    & windeployqt --release --no-translations --no-system-d3d-compiler --no-opengl-sw --no-compiler-runtime $gui
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed" }
    # app-local VC++ runtime for Qt and PAGANI.exe (the command-line tools are linked statically)
    if ($env:VCToolsRedistDir) {
        $crt = Get-ChildItem -Directory (Join-Path $env:VCToolsRedistDir "x64") -Filter "Microsoft.VC*.CRT" | Select-Object -First 1
        if ($crt) { Copy-Item (Join-Path $crt.FullName "*.dll") $stage }
    }
    if (-not (Test-Path (Join-Path $stage "msvcp140.dll"))) {
        Write-Warning "VC++ runtime DLLs not bundled: PAGANI.exe needs the Visual C++ 2015-2022 redistributable"
    }
}
Copy-Item README.md, README_zh.md, CHANGELOG.md, LICENSE -Destination $stage -ErrorAction SilentlyContinue
$docs = Join-Path $stage "docs"
New-Item -ItemType Directory -Force -Path $docs | Out-Null
Copy-Item docs\USER_GUIDE*.md, docs\VALIDATION.md, docs\VALIDATION_REPORT.md -Destination $docs -ErrorAction SilentlyContinue
Copy-Item legacy\doc\PAGANI_manuscript_revised.pdf -Destination $docs -ErrorAction SilentlyContinue

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Created $zip"
