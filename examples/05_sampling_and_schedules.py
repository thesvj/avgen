"""Denoise a latent with the sampler stack, and see why the schedule matters.

Two things decide sample quality as much as the weights do, and both are cheap
to get wrong:

* **The sigma schedule**, and specifically its *shift*. Rectified flow puts a
  fixed budget of steps along the path from noise to data. A 100k-token video
  needs far more of that budget spent at high noise than a 4k-token image does,
  or global structure never resolves. avgen makes the shift a function of the
  sequence length rather than a constant you are expected to retune.
* **The sampler**, which decides how each step integrates the ODE.

Run:
    python examples/05_sampling_and_schedules.py
"""

from __future__ import annotations

import torch

from avgen.infer.sampler import SamplerConfig, build_sampler, list_samplers
from avgen.infer.schedule import (
    ScheduleConfig,
    build_sigma_schedule,
    list_sigma_schedules,
)


def main() -> None:
    print("samplers: ", ", ".join(list_samplers()))
    print("schedules:", ", ".join(list_sigma_schedules()))

    # The same 30-step budget, spent differently depending on sequence length.
    # The anchors here are the ones the shipped multi-node configs train with;
    # a schedule must reproduce whatever the checkpoint was trained under, which
    # is why these are config, not constants.
    def schedule_for(sequence_length: int):
        return build_sigma_schedule(
            ScheduleConfig(
                name="linear",
                steps=30,
                dynamic_shift=True,
                base_length=256,
                base_shift=0.5,
                max_length=131_072,
                max_shift=3.0,
            ),
            sequence_length=sequence_length,
        )

    print()
    for tokens, label in (
        (4_096, "one image"),
        (65_536, "five seconds of 480p"),
        (131_072, "ten seconds of 720p"),
    ):
        schedule = schedule_for(tokens)
        above = int((schedule.sigmas > 0.5).sum())
        print(
            f"{tokens:>7,} tokens ({label:<21}): shift {schedule.shift:.2f}, "
            f"{above}/30 steps above sigma 0.5"
        )
    print(
        "\nThe longer sequence spends more of the same budget at high noise, which\n"
        "is where global structure is decided. A schedule calibrated on images\n"
        "leaves a long clip's structure unformed — this is not a small effect."
    )

    # A full denoising loop. The stand-in "model" below returns the exact
    # velocity for a known straight path, so the trajectory is checkable rather
    # than merely plausible: a correct sampler lands on the target.
    torch.manual_seed(0)
    target = torch.ones(1, 4, 4, 8, 8)

    def denoise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Velocity along the straight path from noise to `target`."""
        return (x - target) / max(sigma, 1e-6)

    sigmas = build_sigma_schedule(
        ScheduleConfig(name="linear", steps=20)
    ).sigmas.tolist()

    print("\nsampler        model calls/step   final error")
    for name in ("euler", "heun", "dpmpp_2m", "res_multistep"):
        sampler = build_sampler(SamplerConfig(name=name))
        sampler.reset()
        torch.manual_seed(0)
        latents = target + torch.randn_like(target)
        for index in range(len(sigmas) - 1):
            latents = sampler.step(
                denoise(latents, sigmas[index]),
                latents,
                sigmas[index],
                sigmas[index + 1],
                # Higher-order samplers evaluate the model again at the
                # provisional endpoint; they need the callback, not just the
                # prediction, and say so rather than silently degrading.
                denoise=denoise,
            )
        error = (latents - target).abs().max().item()
        print(f"  {name:<13}{sampler.evaluations_per_step:^17}{error:.2e}")

    print(
        "\nHigher-order samplers buy accuracy per step with extra model calls. At\n"
        "video sequence lengths a model call is essentially the entire cost of a\n"
        "step, so that trade is rarely free — measure it before assuming it wins."
    )


if __name__ == "__main__":
    main()
