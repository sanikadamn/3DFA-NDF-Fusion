#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p checkpoints
target=checkpoints/multi_category_weights.pth
if [[ -e "$target" ]]; then echo "Keeping existing $target"; exit 0; fi
# Official URL from the uploaded ndf_robot/scripts/download_demo_weights.sh.
curl --fail --location --retry 3 \
  'https://www.dropbox.com/s/hm4hty56ldu1wb5/multi_category_weights.pth?dl=1' \
  --output "${target}.part"
mv "${target}.part" "$target"
echo 'Run python -m tools.check_ndf --device cuda to verify the downloaded weights.'
