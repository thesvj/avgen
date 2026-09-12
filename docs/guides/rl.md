# Reinforcement learning post-training

RL post-training optimises a generator against a reward instead of against a
data distribution. For flow-matching video models it is how you improve the
things a likelihood objective does not measure: prompt adherence, physical
plausibility, aesthetic preference, temporal stability.

Treat it as a finishing step. RL on a weak base model amplifies its failure
modes; it does not fix them.

## Flow-GRPO

```python
from avgen.rl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    state,
    GRPOConfig(
        group_size=8,
        clip_ratio=0.2,
        kl_coefficient=0.02,
        sde_noise_scale=1.0,
        mixgrpo_window=None,
    ),
    reward=reward_model,
    parallel=parallel,
)
```

### Why the ODE becomes an SDE

A trained flow model samples by integrating a deterministic ODE. A deterministic
sampler has no policy to explore with: the same prompt and seed give the same
trajectory, so there is no distribution over actions to compute a ratio against.

Flow-GRPO converts the ODE into an equivalent SDE whose marginals match. That
gives you a stochastic policy — the noise injected at each step is the action —
while leaving the model's learned distribution intact. `sde_noise_scale`
controls the exploration; too little and the group is degenerate, too much and
samples leave the manifold the reward model was calibrated on.

### Group-relative advantage

For each prompt, sample `group_size` trajectories, score them, and use the
group's mean as the baseline. No value network, which is the main practical
appeal: a value head over video latents would be its own training problem.

`group_size=8` is a reasonable floor. Below 4, the baseline is too noisy and the
advantage estimate is mostly variance.

### Ratio clipping and KL

`clip_ratio` bounds the policy update per step, as in PPO. `kl_coefficient`
penalises drift from the reference (pre-RL) model.

**Both are load-bearing.** Without the KL term the model reliably finds the
reward model's blind spots — saturated colour, a frozen scene that scores well
on "consistency", a caption-matching artifact — and quality collapses while the
reward climbs. If your reward is rising and your samples are getting worse, the
KL coefficient is too low. That is the single most common failure of this
method.

### MixGRPO

`mixgrpo_window` restricts the policy-gradient update to a sliding window of
denoising timesteps rather than the whole trajectory. Fewer steps carry
gradients, so an iteration is cheaper, and the window moves over training so
every region is eventually optimised. Use it when the full-trajectory update
does not fit or is too slow.

## Diffusion-DPO

```python
from avgen.rl import DPOConfig, DPOObjective
```

When you have pairwise preferences rather than a scalar reward, flow-DPO
optimises the preference directly and needs no sampling loop during training.
It is cheaper and more stable than GRPO, and limited by the fact that
preferences are fixed: it cannot explore beyond the pairs you collected.

Rule of thumb: DPO if you have a preference dataset, GRPO if you have a reward
model you trust.

## Rewards

```python
from avgen.rl import register_reward


@register_reward("my_reward")
class MyReward:
    def score(self, media, prompts): ...
```

Composite weighted rewards are supported, and usually necessary — a single
reward is a single thing to overfit. A typical mix is prompt adherence plus
aesthetic preference plus a temporal-stability term, with the last one acting as
a regulariser against the "freeze the scene" degenerate solution.

**Log the reward components separately.** An aggregate reward that goes up while
one component collapses is the normal way this fails, and it is invisible in the
total.

## Cost, and how to survive it

RL post-training is expensive because it *generates* during training. Each
iteration is `group_size` full sampling trajectories per prompt, each of which is
a multi-step denoising loop.

What helps, in order:

1. **Fewer sampling steps** during RL than at release time. 10–20 is usually
   enough for the reward to be meaningful.
2. **Lower resolution** during RL, then a short supervised finetune at full
   resolution. The reward signal is mostly resolution-independent.
3. **MixGRPO windows**, so only part of the trajectory carries gradients.
4. **LoRA instead of full-parameter RL.** Much less optimizer state, and the
   adapter is easy to discard when a reward turns out to be badly specified —
   which it often does.

Parallelism is unchanged: the sampling loop uses the same `ModelInput` and the
same context-parallel sharding as training, so a plan that works for training
works here.

## Checklist before starting

- [ ] The base model is good. RL amplifies, it does not repair.
- [ ] The reward model has been checked *against your own generations*, not just
      its validation set.
- [ ] A KL coefficient is set, and you have a reference checkpoint.
- [ ] Reward components are logged individually.
- [ ] Fixed evaluation prompts are sampled every N steps and looked at by a
      human. The reward curve alone will not tell you the model is collapsing.

## Further reading

- [Finetuning](finetuning.md)
- [Inference](inference.md)
