#!/usr/bin/env bash
# Run this on a fresh cluster BEFORE the first training job.
#
#   ./scripts/preflight.sh                 # single node
#   sbatch --nodes 2 scripts/preflight.sh  # two nodes, via SLURM
#
# It answers, in order, the questions that otherwise get answered at 3am:
#
#   1. Does torch see the GPUs, and is it built for their architecture?
#      A cu124 wheel on a B200 silently lacks sm_100 and runs nothing.
#   2. What is the real topology? NVLink domain size decides the tensor-parallel
#      cap, and a wrong assumption there is worth a large fraction of throughput.
#   3. Does the plan the simulator recommends actually apply on this hardware?
#   4. Do the GPU tests pass?
#   5. What is the achieved collective bandwidth? Until this is measured, every
#      time estimate in avgen is a guess rather than a projection.
#
# Nothing here trains. It is cheap, and it is the difference between a first job
# that works and a first week that does not.

set -uo pipefail

PY="${PYTHON:-python}"
FAILED=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
warn() { printf '   ! %s\n' "$1"; FAILED=1; }

step "1. torch and device architecture"
$PY - <<'PYEOF' || warn "torch could not enumerate devices"
import torch
print(f"   torch            {torch.__version__}")
print(f"   cuda available   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("   ! no CUDA device visible")
print(f"   device count     {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"   device 0         {name}  sm_{cap[0]}{cap[1]}")
print(f"   built for        {', '.join(archs)}")
# A device sm_XY runs binaries built for sm_XZ whenever Z <= Y within the same
# major version, and can JIT from any lower PTX. So an exact-match check
# false-alarms on very common setups — an Ada card (sm_89) against a wheel that
# ships sm_86 is fine. What is NOT fine is no compatible binary at all, which is
# the case that wastes a day: the job starts, every kernel falls back or fails,
# and nothing in the error mentions the wheel.
tag = f"sm_{cap[0]}{cap[1]}"
compatible = [
    a for a in archs
    if a.startswith("sm_")
    and a[3:].isdigit()
    and int(a[3:-1] or 0) == cap[0]
    and int(a[-1]) <= cap[1]
]
if tag in archs:
    print(f"   arch match       {tag} (exact)")
elif compatible:
    print(f"   arch match       {tag} via {max(compatible)} (minor-version compatible)")
else:
    raise SystemExit(
        f"   ! this torch build has no binary compatible with {tag}. "
        f"It ships {', '.join(archs)}. Install a matching wheel "
        "(CUDA 12.8+ for Blackwell / sm_100) before training."
    )
free, total = torch.cuda.mem_get_info(0)
print(f"   memory           {total / 1024**3:.0f} GiB total, {free / 1024**3:.0f} GiB free")
PYEOF

step "2. interconnect topology"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi topo -m 2>/dev/null || warn "nvidia-smi topo unavailable"
else
    warn "nvidia-smi not found"
fi

step "3. does the recommended plan apply on this hardware"
$PY -m avgen.cli.main info || warn "avgen info failed"
GPUS="$($PY -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')"
$PY -m avgen.cli.main plan \
    --world-size "${GPUS}" --seq-len 16384 --params 2e9 --depth 32 --width 2560 \
    || warn "avgen plan failed"

step "4. GPU tests"
$PY -m pytest -m "gpu and not multigpu" -q || warn "single-device GPU tests failed"
if [ "${GPUS}" -ge 2 ]; then
    torchrun --standalone --nproc-per-node 2 -m pytest -m multigpu -q \
        || warn "multi-GPU tests failed"
else
    printf '   (skipping multigpu: only %s device visible)\n' "${GPUS}"
fi

step "5. collective bandwidth — calibrates every time estimate"
cat <<'EOF'
   avgen ships plausible fabric defaults, not measurements. Measure yours once:

     git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make
     ./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8        # intra-node
     srun -N2 --ntasks-per-node 8 ./build/all_reduce_perf -b 1G -e 8G -f 2

   Then feed the reported busbw back in, once:

     from avgen.simulate import calibrate_from_busbw
     nvlink = calibrate_from_busbw("NVLink", measured_busbw_gbps=372.0, peak_gbps=450.0)

   Until you do, treat every step-time and MFU number avgen prints as an
   estimate with unquantified error.
EOF

if [ "$FAILED" -eq 0 ]; then
    printf '\n\033[1mpreflight: clean. Safe to launch a real job.\033[0m\n'
else
    printf '\n\033[1mpreflight: issues above must be resolved before training.\033[0m\n'
fi
exit "$FAILED"
