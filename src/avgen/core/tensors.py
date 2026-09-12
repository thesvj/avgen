"""Portable dtype names and the auxiliary-target tensor bundle.

A generative video trainer accumulates auxiliary supervision over its lifetime:
optical-flow targets, depth, segmentation, a reward score, a progress label. If
each of those is bolted onto the batch dataclass, every extension is a breaking
change to a structure that crosses the ``torch.compile`` boundary and gets
serialised into checkpoints.

:class:`TensorBundle` is the alternative: a fixed tuple of tensors whose meaning
is owned by a separate, JSON-serialisable :class:`TensorBundleSpec`. Research
code adds a target by adding a spec entry; the batch type never changes and the
shard format stays forward compatible.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self, cast

import torch
from torch.utils import _pytree

from avgen.core._validate import require_name, require_unique_names

__all__ = ["TensorBundle", "TensorBundleSpec", "TensorDType"]


class TensorDType(StrEnum):
    """Portable dtype names admitted by auxiliary tensor bundles.

    The bundle spec has to survive a JSON round trip through a checkpoint
    manifest and a shard header, so dtypes are named rather than pickled.
    """

    BOOL = "bool"
    INT32 = "int32"
    INT64 = "int64"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the corresponding PyTorch dtype."""
        return _TORCH_DTYPES[self]

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> TensorDType:
        """Return the portable name for a PyTorch dtype.

        Args:
            dtype: A PyTorch dtype.

        Returns:
            The matching :class:`TensorDType`.

        Raises:
            KeyError: If the dtype has no portable name.
        """
        for name, candidate in _TORCH_DTYPES.items():
            if candidate is dtype:
                return name
        raise KeyError(
            f"dtype {dtype} has no portable name; "
            f"supported: {', '.join(sorted(member.value for member in cls))}"
        )


_TORCH_DTYPES: dict[TensorDType, torch.dtype] = {
    TensorDType.BOOL: torch.bool,
    TensorDType.INT32: torch.int32,
    TensorDType.INT64: torch.int64,
    TensorDType.BF16: torch.bfloat16,
    TensorDType.FP16: torch.float16,
    TensorDType.FP32: torch.float32,
}


class TensorBundleSpec:
    """Static names, meanings, shapes, and dtypes for an auxiliary tensor tuple.

    The spec is immutable, hashable, and JSON-serialisable. It travels with the
    batch through ``torch.compile`` as static metadata, which means a change of
    auxiliary targets forces a recompile rather than silently reinterpreting
    tensors.

    Args:
        names: Field names, unique and trimmed.
        semantics: One free-text meaning per field, for the manifest and docs.
        shapes: One concrete shape per field.
        dtypes: One portable dtype per field.

    Raises:
        ValueError: If the four sequences differ in length, if names repeat, or
            if any shape contains a negative dimension.
    """

    __slots__ = ("_dtypes", "_names", "_semantics", "_shapes")

    def __init__(
        self,
        names: tuple[str, ...] = (),
        semantics: tuple[str, ...] = (),
        shapes: tuple[tuple[int, ...], ...] = (),
        dtypes: tuple[TensorDType, ...] = (),
    ) -> None:
        lengths = {len(names), len(semantics), len(shapes), len(dtypes)}
        if len(lengths) != 1:
            raise ValueError(
                "tensor bundle spec fields must have equal lengths; got "
                f"names={len(names)}, semantics={len(semantics)}, "
                f"shapes={len(shapes)}, dtypes={len(dtypes)}"
            )
        require_unique_names("names", names)
        for index, semantic in enumerate(semantics):
            require_name(f"semantics[{index}]", semantic)
        for field_index, shape in enumerate(shapes):
            for dim_index, dimension in enumerate(shape):
                if isinstance(dimension, bool) or dimension < 0:
                    raise ValueError(
                        "tensor bundle shapes must contain non-negative integers; "
                        f"got shapes[{field_index}][{dim_index}]={dimension!r}"
                    )
        self._names = names
        self._semantics = semantics
        self._shapes = shapes
        self._dtypes = dtypes

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in tuple order."""
        return self._names

    @property
    def semantics(self) -> tuple[str, ...]:
        """Free-text meaning of each field."""
        return self._semantics

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Concrete shape of each field."""
        return self._shapes

    @property
    def dtypes(self) -> tuple[TensorDType, ...]:
        """Portable dtype of each field."""
        return self._dtypes

    def __len__(self) -> int:
        """Return the number of fields in the bundle."""
        return len(self._names)

    def __eq__(self, other: object) -> bool:
        """Compare specs field by field."""
        if not isinstance(other, TensorBundleSpec):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash the spec so it can key a compile cache."""
        return hash(self._key())

    def __repr__(self) -> str:
        """Return a compact developer representation."""
        fields = ", ".join(
            f"{name}:{dtype.value}{list(shape)}"
            for name, dtype, shape in zip(
                self._names, self._dtypes, self._shapes, strict=True
            )
        )
        return f"TensorBundleSpec({fields})"

    def _key(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[TensorDType, ...],
    ]:
        return (self._names, self._semantics, self._shapes, self._dtypes)

    def index_of(self, name: str) -> int:
        """Return the tuple position of a named field.

        Args:
            name: Field name.

        Returns:
            The zero-based position of the field.

        Raises:
            KeyError: If the spec has no such field.
        """
        try:
            return self._names.index(name)
        except ValueError as error:
            available = ", ".join(self._names) or "(none)"
            raise KeyError(
                f"unknown bundle field {name!r}; available: {available}"
            ) from error

    def with_batch_size(self, batch: int) -> TensorBundleSpec:
        """Return a copy whose leading dimension is ``batch``.

        Args:
            batch: New leading dimension.

        Returns:
            A spec identical apart from the batch dimension of every field.
        """
        return TensorBundleSpec(
            names=self._names,
            semantics=self._semantics,
            shapes=tuple((batch, *shape[1:]) for shape in self._shapes),
            dtypes=self._dtypes,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "names": list(self._names),
            "semantics": list(self._semantics),
            "shapes": [list(shape) for shape in self._shapes],
            "dtypes": [dtype.value for dtype in self._dtypes],
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Self:
        """Restore a spec from its JSON-safe representation.

        Args:
            values: Mapping produced by :meth:`to_dict`.

        Returns:
            The restored spec.
        """
        return cls(
            names=tuple(cast("list[str]", values["names"])),
            semantics=tuple(cast("list[str]", values["semantics"])),
            shapes=tuple(
                tuple(shape) for shape in cast("list[list[int]]", values["shapes"])
            ),
            dtypes=tuple(
                TensorDType(value) for value in cast("list[str]", values["dtypes"])
            ),
        )


class TensorBundle:
    """A fixed tensor tuple whose interpretation is owned by a bundle spec.

    Args:
        values: The tensors, in spec order.
    """

    __slots__ = ("values",)

    def __init__(self, values: tuple[torch.Tensor, ...] = ()) -> None:
        self.values = values

    def __len__(self) -> int:
        """Return the number of tensors held."""
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        """Compare bundles by tensor identity, not by value."""
        if not isinstance(other, TensorBundle):
            return NotImplemented
        return self.values is other.values or all(
            left is right
            for left, right in zip(self.values, other.values, strict=False)
        )

    def __repr__(self) -> str:
        """Return a compact developer representation."""
        return f"TensorBundle({len(self.values)} tensors)"

    def get(self, spec: TensorBundleSpec, name: str) -> torch.Tensor:
        """Return one named tensor.

        Args:
            spec: The spec that interprets this bundle.
            name: Field name.

        Returns:
            The tensor stored at the named position.

        Raises:
            KeyError: If the spec has no such field.
            IndexError: If the bundle is shorter than the spec.
        """
        return self.values[spec.index_of(name)]

    def validate(
        self,
        spec: TensorBundleSpec,
        *,
        device: torch.device | None = None,
    ) -> None:
        """Validate tuple length, shapes, dtypes, and optional device.

        Args:
            spec: The spec that interprets this bundle.
            device: Device every tensor must live on, when given.

        Raises:
            ValueError: On a length, shape, or device mismatch.
            TypeError: On a dtype mismatch.
        """
        if len(self.values) != len(spec):
            raise ValueError(
                "tensor bundle length must match its spec; got "
                f"{len(self.values)} values for {len(spec)} fields"
            )
        for index, (value, shape, dtype) in enumerate(
            zip(self.values, spec.shapes, spec.dtypes, strict=True)
        ):
            name = spec.names[index]
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"target {name!r} shape must be {shape}; got {tuple(value.shape)}"
                )
            if value.dtype is not dtype.torch_dtype:
                raise TypeError(
                    f"target {name!r} dtype must be {dtype.torch_dtype}; "
                    f"got {value.dtype}"
                )
            if device is not None and value.device != device:
                raise ValueError(
                    f"target {name!r} must be on device {device}; got {value.device}"
                )


def _flatten_bundle(
    bundle: TensorBundle,
) -> tuple[list[torch.Tensor], None]:
    return list(bundle.values), None


def _unflatten_bundle(
    values: list[torch.Tensor],
    _context: None,
) -> TensorBundle:
    return TensorBundle(tuple(values))


_pytree.register_pytree_node(
    TensorBundle,
    _flatten_bundle,
    _unflatten_bundle,
    serialized_type_name="avgen.TensorBundle",
)
