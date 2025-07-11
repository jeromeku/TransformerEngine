#/bin/bash

set -euo pipefail
DEBUG_LOG="dl_debug.log"
CMD="LD_DEBUG=libs python test_cdll.py"
echo $CMD

eval $CMD 2>&1 | tee ${DEBUG_LOG}
echo "Debug log: ${DEBUG_LOG}"