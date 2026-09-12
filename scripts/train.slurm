#!/bin/bash
# Multi-node avgen training under SLURM.
#
#   sbatch --nodes 8 scripts/train.slurm configs/train/multinode_64.yaml
#   sbatch --nodes 8 scripts/train.slurm configs/train/multinode_64.yaml parallel.context=8
#
# Anything after the config path is passed through as a dotted config override.
#
# The parts that are not boilerplate, and why:
#
#   * One task per NODE, not per GPU. torchrun forks the per-GPU processes
#     itself; letting SLURM do it too gives you world_size^2 processes, which
#     usually presents as a rendezvous that never completes.
#   * The rendezvous host is derived from SLURM's own node list rather than
#     hardcoded, so the script does not care which nodes you are given.
#   * A free rendezvous port is derived from the job ID. A fixed port collides
#     the moment two of your jobs land on the same node, and the failure looks
#     like a hang rather than a bind error.
#   * NCCL debug output is INFO on rank 0 only. At 1024 ranks, WARN on every
#     rank is already more log than anyone reads.

#SBATCH --job-name=avgen
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/avgen-%j.out
#SBATCH --error=logs/avgen-%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/train.slurm <config.yaml> [key=value ...]}"
shift || true
OVERRIDES=("$@")

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"

# Rendezvous on the first node in the allocation.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
# Derive a port from the job id so concurrent jobs never collide.
MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"

export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# Flight recorder: the only practical way to diagnose a hang at scale. It costs
# a small ring buffer per rank and tells you which collective on which rank
# never completed. avgen sets these too, but exporting here covers any
# subprocess that starts before the framework is imported.
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS=INIT,ENV

mkdir -p logs

echo "avgen: ${NNODES} nodes x ${GPUS_PER_NODE} gpus = $(( NNODES * GPUS_PER_NODE )) ranks"
echo "avgen: rendezvous ${MASTER_ADDR}:${MASTER_PORT}"
echo "avgen: config ${CONFIG} ${OVERRIDES[*]:-}"

# --rdzv-backend=c10d with a static endpoint: torchelastic then handles a failed
# rank by restarting the group rather than killing the job.
srun --kill-on-bad-exit=1 torchrun \
    --nnodes "${NNODES}" \
    --nproc-per-node "${GPUS_PER_NODE}" \
    --rdzv-backend c10d \
    --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id "${SLURM_JOB_ID}" \
    --max-restarts 3 \
    -m avgen.cli.main train \
    --config "${CONFIG}" \
    "${OVERRIDES[@]}"
