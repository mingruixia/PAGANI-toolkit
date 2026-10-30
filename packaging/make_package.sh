#!/bin/bash
# Assemble the Linux command-line package from a CMake build directory.
#   packaging/make_package.sh [build_dir] [out_dir]
set -euo pipefail
build=${1:-build}
out=${2:-dist}
version=$(sed -n 's/^project(PAGANI_Toolkit VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)
name="PAGANI-Toolkit-$version-linux-x64"
stage="$out/$name"
rm -rf "$stage"
mkdir -p "$stage/docs"
cp -r "$build/package/exefiles" "$build/package/exefiles_weighted" "$stage/"
cp README.md README_zh.md CHANGELOG.md "$stage/"
[ -f LICENSE ] && cp LICENSE "$stage/"
cp docs/USER_GUIDE*.md docs/VALIDATION.md docs/VALIDATION_REPORT.md "$stage/docs/"
cp legacy/doc/PAGANI_manuscript_revised.pdf "$stage/docs/" 2>/dev/null || true
cp packaging/linux/example_pipeline.sh "$stage/"
chmod +x "$stage/example_pipeline.sh" "$stage"/exefiles/* "$stage"/exefiles_weighted/*
tar -C "$out" -czf "$out/$name.tar.gz" "$name"
echo "Created $out/$name.tar.gz"
