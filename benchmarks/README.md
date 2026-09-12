# Benchmarks

`avgen plan` predicts step time and peak memory analytically, and a prediction
nobody checks is just a guess with a table around it. These scripts measure the
real thing, so the simulator's error stays a known quantity.

```bash
python benchmarks/bench_step.py                              # tiny, CPU, seconds
python benchmarks/bench_step.py --model dit_2b --device cuda # a real measurement
```

## Reading the output

`bench_step.py` reports the **median** step time over the measured steps, and
the fastest separately. Warmup steps are thrown away rather than averaged in.
The first step pays for lazy kernel selection, allocator growth and, under
`torchrun`, NCCL channel setup, so including it understates steady-state
throughput by a factor that depends on how many steps you happened to run.

The script prints the matching `avgen plan` invocation. Do compare the two. If
the gap is more than roughly 20% at a shape the simulator claims to model,
please open an issue. It means either the cost model or the configuration is
wrong, and both are worth knowing about.

## What a benchmark here should do

- Report a median and a spread, never a single number.
- Discard warmup explicitly, and say how many steps it dropped.
- Name the hardware, the shapes and the precision in its own output, so that a
  result pasted into an issue can be understood without the command behind it.
