#!/bin/bash
# Builds libv4l2_capture.so for the current architecture (run this on the
# target — x86_64 dev VM for a compile check, the Q6A board for arm64).
set -euo pipefail
cd "$(dirname "$0")"
gcc -shared -fPIC -Wall -Wextra -O2 -o libv4l2_capture.so v4l2_capture.c
echo "Built $(pwd)/libv4l2_capture.so ($(uname -m))"
