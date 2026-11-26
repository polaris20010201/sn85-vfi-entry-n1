#!/bin/sh
# Competition contract: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR. Always exits 0.
set -u
umask 022
app_dir=${CONTENDER_APP_DIR:-/app}
python=${PYTHON:-python3}
# Pairs per GPU batch for interp_fast.py (its own default is 8); the environment can override it.
export VFI_BATCH="${VFI_BATCH:-2}"
mkdir -p "$2" 2>/dev/null
timeout -k 10 "${MAIN_SECONDS:-3300}" "$python" -I "$app_dir/interp_fast.py" "$1" "$2"
rm -f -- "$2"/.partial-* 2>/dev/null
exit 0
