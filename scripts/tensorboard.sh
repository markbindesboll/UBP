#!/bin/bash
### Serve TensorBoard for the UBP runs on the local workstation.
###
###   /bin/bash scripts/tensorboard.sh [PORT]
###
### Then from your laptop, tunnel to this workstation:
###   ssh -L 6006:localhost:6006 s193209@comp-ws6211
### and open http://localhost:6006
###
### (Includes the PYTHONPATH dance for tensorboard 2.19.0 compatibility)

set -euo pipefail

PORT=${1:-6006}
LOGDIR=${LOGDIR:-/data/thingseeg2/ubp_exp}
TB_PREFIX="$HOME/tb_standalone"
PYBIN="$HOME/miniforge3/envs/ubp/bin/python"

if [ ! -d "$TB_PREFIX" ]; then
    echo "installing standalone tensorboard to $TB_PREFIX"
    "$PYBIN" -m pip install -q --no-deps --target "$TB_PREFIX" "tensorboard==2.19.0"
fi

echo "logdir : $LOGDIR"
echo "host   : $(hostname)"
echo "url    : http://localhost:${PORT} after running this on your laptop:"
echo "         ssh -L ${PORT}:localhost:${PORT} $USER@comp-ws6211"

PYTHONPATH="$TB_PREFIX" exec "$PYBIN" -m tensorboard.main \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --reload_interval 30