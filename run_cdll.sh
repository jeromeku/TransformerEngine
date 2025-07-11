#/bin/bash
# LD_DEBUG=symbols        # show symbol look-ups
# LD_DEBUG=bindings       # lazy PLT fix-ups
# LD_DEBUG=reloc          # every relocation applied
# LD_DEBUG=statistics     # totals at program exit
# LD_DEBUG=help           # print all options

set -euo pipefail
DEBUG_LOG="cddl.trace.log"
LD_DEBUG_OUTPUT=${DEBUG_LOG}

STDOUT_LOG="cdll.log"

LD_FLAGS=libs,statistics

CMD="LD_DEBUG_OUTPUT=${DEBUG_LOG} LD_DEBUG=${LD_FLAGS} python test_cdll.py"
echo $CMD

eval $CMD 2>&1 2>&1 | tee ${STDOUT_LOG}
echo "Trace log: ${DEBUG_LOG}, Stdout log: ${STDOUT_LOG}"