"""Fine-tuning: adapters, parameter selection, and stage recipes.

Everything here operates on a model that is already trained. The package makes
three claims and each one is testable:

* **An adapter starts as an exact identity.** After :func:`apply_lora` or an
  :class:`~avgen.finetune.adapters.AdapterHandles` attachment, the model's output
  is bit-identical to the base model's. Nothing degrades before the first
  optimizer step, so any degradation observed later is attributable.
* **Sharding is not an afterthought.** The adapters derive their DTensor
  placements from the base weights and refuse to attach where they would
  silently diverge. See :mod:`avgen.finetune.lora` for the composition order.
* **A recipe is data.** :class:`~avgen.finetune.stages.FinetuneStage` is a value
  that can be serialised, diffed and logged, rather than a training script.

Typical use::

    from avgen.finetune import LoRAConfig, apply_lora, mark_only_lora_trainable
    from avgen.parallel import FSDPConfig, ParallelConfig, parallelize

    apply_lora(model, LoRAConfig(rank=32, alpha=64.0))
    mark_only_lora_trainable(model)
    parallel = parallelize(
        model, dims,
        config=ParallelConfig(fsdp=FSDPConfig(ignore_frozen_params=True)),
    )
"""

from avgen.finetune.adapters import (
    AdapterHandles,
    ControlAdapter,
    ControlAdapterConfig,
    IPAdapter,
    IPAdapterConfig,
)
from avgen.finetune.freeze import (
    TrainableSummary,
    freeze_all,
    freeze_except,
    freeze_matching,
    trainable_summary,
)
from avgen.finetune.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    apply_lora,
    load_adapter,
    lora_modules,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora,
    save_adapter,
    unmerge_lora,
)
from avgen.finetune.stages import (
    DistillationConfig,
    FinetuneStage,
    TimestepPlan,
    apply_stage,
    get_stage,
    list_stages,
    register_stage,
    resolution_shift,
    shift_timesteps,
    stage_names_for,
    standard_stages,
)

__all__ = [
    "AdapterHandles",
    "ControlAdapter",
    "ControlAdapterConfig",
    "DistillationConfig",
    "FinetuneStage",
    "IPAdapter",
    "IPAdapterConfig",
    "LoRAConfig",
    "LoRALinear",
    "TimestepPlan",
    "TrainableSummary",
    "adapter_state_dict",
    "apply_lora",
    "apply_stage",
    "freeze_all",
    "freeze_except",
    "freeze_matching",
    "get_stage",
    "list_stages",
    "load_adapter",
    "lora_modules",
    "lora_parameters",
    "mark_only_lora_trainable",
    "merge_lora",
    "register_stage",
    "resolution_shift",
    "save_adapter",
    "shift_timesteps",
    "stage_names_for",
    "standard_stages",
    "trainable_summary",
    "unmerge_lora",
]
