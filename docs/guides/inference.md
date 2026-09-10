# Inference

Inference shares `ModelInput` with training. The same tokens, the same physical
coordinates, the same condition modes, the same patchifier. There is no separate
inference-time model definition to drift out of sync with the trained one — a
class of bug that is otherwise very hard to find, because the model appears to
work and simply generates slightly wrong things.

## Generate

```python
from avgen.infer import GenerationPipeline, GuidanceConfig, SamplerConfig

pipeline = GenerationPipeline(model, video_codec=codec, text_encoder=encoder)
media = pipeline(
    prompts=["a paper boat drifting down a rain gutter, close up"],
    steps=30,
    sampler=SamplerConfig(name="dpmpp_2m"),
    guidance=GuidanceConfig(scale=4.5, rescale=0.7),
    seed=0,
)
```

```bash
avgen generate --checkpoint runs/exp/latest \
  --prompt "..." --steps 30 --guidance 4.5 --seed 0 --output samples/
```

## Samplers

| Name | Order | When |
|---|---|---|
| `euler` | 1 | The baseline. Correct, needs the most steps. Use it to check a new sampler. |
| `heun` | 2 | Two model evaluations per step; better per *step*, not per *evaluation*. |
| `dpmpp_2m` | 2, multistep | Reuses the previous evaluation, so second-order accuracy at first-order cost. The usual default. |
| `res_multistep` | 2+ | Exponential-integrator variant; competitive at low step counts. |

For rectified flow, 25–40 steps is the normal range. Below about 15 the samples
degrade in a characteristic way — motion becomes smooth and detail becomes
plastic — because the ODE is being integrated too coarsely, not because the
model is weak.

**When comparing samplers, compare at equal model evaluations, not equal steps.**
Heun at 20 steps is 40 evaluations and should be compared against Euler at 40.

## Guidance

```python
GuidanceConfig(scale=4.5, rescale=0.7, audio_scale=3.0)
```

- **`scale`** — classifier-free guidance. 3–6 is the usual window for video.
  Higher gives stronger prompt adherence and more saturation, contrast crush,
  and motion artifacts.
- **`rescale`** — CFG-rescale, which corrects the over-exposure that high
  guidance introduces by rescaling the guided prediction back toward the
  conditional prediction's statistics. 0.5–0.7 lets you use a higher `scale`
  without the wash-out.
- **`text_scale` / `video_scale` / `audio_scale`** — separate guidance strength per
  conditioning source, composed multiplicatively, for AV models.
  Audio typically wants less guidance than video; a single shared scale
  over-drives one of them.
- **APG** — adaptive projected guidance, which projects out the component of the
  guidance update that only increases magnitude. Useful at high guidance scales
  where CFG-rescale alone is not enough.

CFG costs a second forward pass per step. `ModelInput.unconditional()` produces
the null-conditioned input, so the two passes can be batched.

## Conditioning modes

The same `ConditionMode` values used in training drive inference:

| Mode | What you provide |
|---|---|
| `VIDEO_ONLY` | Text only — plain text-to-video |
| `IMAGE_TO_VIDEO` | A first frame |
| `CONTINUATION` | Trailing frames of an existing clip |
| `INPAINT` | A masked region to fill |
| `VIDEO_TO_VIDEO` | A source clip to transform |
| `JOINT` | Generate video and audio together |
| `VIDEO_TO_AUDIO` / `AUDIO_TO_VIDEO` | One modality conditions the other |

Conditioned tokens are marked clean anchors in `TokenStream.conditioned`: noise
level zero, excluded from the update, and the mask survives context-parallel
sharding. One code path covers every mode.

## Long clips: context parallelism at inference

A 10-second 720p generation is 216,000 tokens. That does not fit on one GPU, for
the same reason it does not fit during training, and the fix is the same:

```python
dims = ParallelDims(world_size=8, context=8)
```

The sampler runs inside a context-parallel region and the sequence is gathered
once at the end, before the codec decodes it. Note that inference is
*latency*-bound rather than throughput-bound, so the ring communication is
harder to hide — there is no backward pass to overlap it with. Expect worse
scaling efficiency than training at the same `cp`.

## Determinism

Same seed, same config, same rank count, same result. `RNGStreams` keeps the
sampler's generator separate from the training streams, so changing the training
seed does not change your evaluation samples.

Changing the rank count changes the result, because the sharding changes the
order of floating-point reductions. That is expected. Fix the rank count when
you are comparing checkpoints.

## Decode

`GeneratedMedia` holds latents. Decoding to pixels needs a codec, and therefore
`avgen[codecs]`. Keeping decode separate means you can generate on the cluster
and decode elsewhere, and it keeps `diffusers` off the generating ranks.

## Performance notes

- **bf16 throughout.** fp16 has no advantage here and a much smaller range.
- **`torch.compile` the blocks.** The sampler calls the same graph 30 times, so
  compilation amortises immediately.
- **Batch prompts.** Sampling is memory-bandwidth-bound at small batch; four
  prompts at once is often barely slower than one.
- **Cache the text encoding.** Encoding a prompt is a full text-tower forward;
  do it once for a sweep over seeds or guidance scales.

## Further reading

- [Reinforcement learning](rl.md)
- [Context parallelism for video](context-parallel-for-video.md)
