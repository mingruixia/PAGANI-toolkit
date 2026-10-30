#!/bin/bash
# Build environment of the Linux package, on top of the CUDA image
# nvidia/cuda:12.6.3-devel-rockylinux8 (used by CI and for local builds).
# Rocky Linux 8 has glibc 2.28, so the tools also run on RHEL/Rocky 8,
# Ubuntu 20.04, Debian 11 and newer.  gcc-toolset-12 links the C++ library
# parts newer than the system's GCC 8 statically, so the binaries need only
# the system libstdc++ (and the NVIDIA driver).
#   source packaging/linux/setup_build_env.sh   (installs, then enables the toolset)
set -e
if ! [ -x /opt/rh/gcc-toolset-12/root/usr/bin/g++ ]; then
  dnf install -y -q gcc-toolset-12-gcc-c++ python3.11 python3.11-pip git tar gzip findutils >/dev/null
  python3.11 -m pip install -q --upgrade pip
  python3.11 -m pip install -q cmake==3.31.6 ninja==1.11.1.4
  python3.11 -m pip install -q -r "$(dirname "${BASH_SOURCE[0]}")/../../tests/requirements.txt"
fi
source /opt/rh/gcc-toolset-12/enable
export PATH="/usr/local/bin:$PATH"
set +e
