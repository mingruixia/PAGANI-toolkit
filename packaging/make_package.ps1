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
}
Copy-Item README.md, LICENSE -Destination $stage -ErrorAction SilentlyContinue
Copy-Item -Recurse docs (Join-Path $stage "docs")
Copy-Item -Recurse scripts (Join-Path $stage "scripts") -ErrorAction SilentlyContinue

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Created $zip"
