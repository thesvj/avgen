"""Model registry: name to architecture, including third-party architectures.

A config file names a model with a string. Something has to turn that string
into a class, and the two usual answers are both wrong at the scale this
framework targets. A hard-coded ``if/elif`` chain means every new architecture
patches a central file, which makes forks diverge immediately. Importing by
dotted path means a config file can execute arbitrary code, and a typo produces
an ``ImportError`` about a module the user never mentioned.

The registry is an explicit table, populated two ways:

* **In-tree**, by the :func:`register_model` decorator at class definition
  time. Importing :mod:`avgen.models` is what registers the built-ins.
* **Out-of-tree**, by ``importlib.metadata`` entry points in the group
  ``avgen.models``. A third party ships their architecture in their own package
  with::

      [project.entry-points."avgen.models"]
      my_dit = "my_package.model:MyDiT"

  and ``avgen train --config ... model.name=my_dit`` finds it with no patch to
  avgen and no dotted path in the config file. Discovery is lazy — it costs a
  metadata scan, so it happens on the first lookup miss rather than at import.

Every model class is expected to take exactly one constructor argument, a frozen
config dataclass, and the registry reads that dataclass off the annotation. That
is what lets :func:`build_model` accept a plain mapping from a YAML file and
still get validation, defaults, and a precise error naming the offending field.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from importlib import metadata
from typing import Any, TypeVar, get_type_hints

from torch import nn

__all__ = [
    "MODEL_ENTRY_POINT_GROUP",
    "build_model",
    "list_models",
    "model_config_class",
    "register_model",
]

#: Entry-point group third-party packages advertise their architectures in.
MODEL_ENTRY_POINT_GROUP = "avgen.models"

_MODELS: dict[str, type[nn.Module]] = {}
_CONFIGS: dict[str, type] = {}
_entry_points_loaded = False

_ModelT = TypeVar("_ModelT", bound=nn.Module)


def _config_class_for(cls: type) -> type:
    """Infer the config dataclass a model class is constructed from.

    Args:
        cls: The model class.

    Returns:
        The config dataclass type.

    Raises:
        TypeError: If the constructor does not take a single annotated config
            dataclass, which is the whole contract that makes a YAML mapping
            buildable.
    """
    declared = getattr(cls, "config_class", None)
    if isinstance(declared, type):
        return declared

    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    if not parameters:
        raise TypeError(
            f"{cls.__name__} must take a config dataclass as its first "
            "constructor argument, or declare a `config_class` attribute"
        )
    # Annotations are strings under `from __future__ import annotations`, so they
    # have to be resolved against the defining module rather than read directly.
    hints = get_type_hints(cls.__init__)
    config_type = hints.get(parameters[0].name)
    if not isinstance(config_type, type) or not is_dataclass(config_type):
        raise TypeError(
            f"{cls.__name__}.__init__ parameter {parameters[0].name!r} must be "
            f"annotated with a config dataclass; got {config_type!r}"
        )
    return config_type


def register_model(name: str) -> Callable[[type[_ModelT]], type[_ModelT]]:
    """Register a model class under a name.

    Args:
        name: Registry key, as it appears in a config file.

    Returns:
        A class decorator that registers and returns the class unchanged.

    Raises:
        ValueError: If the name is empty, untrimmed, or already taken. A
            duplicate is refused rather than overwritten: silently shadowing an
            architecture would make two configs that name the same model build
            different networks depending on import order.
    """
    if not name or name.strip() != name:
        raise ValueError(f"model name must be a non-empty trimmed string; got {name!r}")

    def decorate(cls: type[_ModelT]) -> type[_ModelT]:
        if name in _MODELS and _MODELS[name] is not cls:
            raise ValueError(
                f"model {name!r} is already registered to "
                f"{_MODELS[name].__module__}.{_MODELS[name].__qualname__}"
            )
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{cls.__name__} must subclass nn.Module to be registered")
        _MODELS[name] = cls
        _CONFIGS[name] = _config_class_for(cls)
        return cls

    return decorate


def _load_entry_points() -> None:
    """Discover third-party models advertised through entry points.

    Raises:
        RuntimeError: If an advertised entry point cannot be loaded. Failing
            loudly beats hiding it: a plugin that silently does not load looks
            exactly like a typo in the model name, hours later.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    # Sorted so that two ranks with the same installed packages register in the
    # same order; unordered iteration here would be a determinism hazard.
    points = sorted(
        metadata.entry_points(group=MODEL_ENTRY_POINT_GROUP), key=lambda ep: ep.name
    )
    for point in points:
        if point.name in _MODELS:
            continue
        try:
            loaded = point.load()
        except Exception as error:
            raise RuntimeError(
                f"failed to load model entry point {point.name!r} "
                f"({point.value}) from group {MODEL_ENTRY_POINT_GROUP!r}"
            ) from error
        # A class decorated with @register_model registers itself on import; one
        # that is not gets registered here under its entry-point name.
        if point.name not in _MODELS:
            if not (isinstance(loaded, type) and issubclass(loaded, nn.Module)):
                raise RuntimeError(
                    f"model entry point {point.name!r} ({point.value}) must resolve "
                    f"to an nn.Module subclass; got {loaded!r}"
                )
            _MODELS[point.name] = loaded
            _CONFIGS[point.name] = _config_class_for(loaded)


def _resolve(name: str) -> type[nn.Module]:
    """Look up a registered model class.

    Args:
        name: Registry key.

    Returns:
        The model class.

    Raises:
        KeyError: If no model is registered under that name.
    """
    if name not in _MODELS:
        _load_entry_points()
    try:
        return _MODELS[name]
    except KeyError:
        available = ", ".join(sorted(_MODELS)) or "<none>"
        raise KeyError(
            f"unknown model {name!r}; registered models: {available}. "
            f"Third-party models must advertise an entry point in the "
            f"{MODEL_ENTRY_POINT_GROUP!r} group."
        ) from None


def list_models() -> tuple[str, ...]:
    """List every registered model name.

    Returns:
        Sorted registry keys, including third-party entry points.
    """
    _load_entry_points()
    return tuple(sorted(_MODELS))


def model_config_class(name: str) -> type:
    """Return the config dataclass a model is built from.

    Args:
        name: Registry key.

    Returns:
        The config dataclass type.

    Raises:
        KeyError: If no model is registered under that name.
    """
    _resolve(name)
    return _CONFIGS[name]


def _coerce(config_class: type, values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a plain mapping into constructor arguments for a config.

    Nested dataclass fields are built from nested mappings, so a YAML file can
    express ``rope_scaling: {mode: ntk, time_factor: 2.0}`` without the loader
    knowing anything about rotary embeddings.

    Args:
        config_class: The target config dataclass.
        values: Raw values, typically straight from YAML.

    Returns:
        Keyword arguments for the config's constructor.

    Raises:
        ValueError: If a key does not name a field of the config.
    """
    known = {field.name for field in fields(config_class)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"unknown config field(s) for {config_class.__name__}: "
            f"{', '.join(unknown)}; known fields: {', '.join(sorted(known))}"
        )
    # `Field.type` is a *string* under PEP 563, so the annotations have to be
    # resolved before a nested dataclass can be recognised.
    hints = get_type_hints(config_class)
    coerced: dict[str, Any] = {}
    for key, value in values.items():
        field_type = hints.get(key)
        if (
            isinstance(value, Mapping)
            and isinstance(field_type, type)
            and is_dataclass(field_type)
        ):
            coerced[key] = field_type(**value)
        else:
            coerced[key] = value
    return coerced


def build_model(name: str, config: Mapping[str, Any]) -> nn.Module:
    """Build a registered model from a plain configuration mapping.

    Build under ``torch.device("meta")`` for anything that will not fit on one
    device, then materialise per shard; see
    :meth:`~avgen.models.dit.VideoDiT.init_weights`.

    Args:
        name: Registry key.
        config: Field values for the model's config dataclass. Values absent
            from the mapping take the dataclass default.

    Returns:
        The constructed model.

    Raises:
        KeyError: If no model is registered under that name.
        ValueError: If a config field is unknown or fails the config's own
            validation.
    """
    model_class = _resolve(name)
    config_class = _CONFIGS[name]
    return model_class(config_class(**_coerce(config_class, config)))
