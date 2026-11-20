"""Partition transforms: deterministic maps from a source value to a partition value.

Each transform is a frozen dataclass whose ``to_dict`` gives the transform object
of a partition spec (``docs/specs/partition_spec.md``), with an ``"@type"`` member
that names it. Each known transform has:

- ``apply(values)``, which maps an integer Arrow array to an array of partition values.
- ``PARTITION_TYPE``, the Arrow type of those values and of the partition column.
  Each transform spec defines that type.
- ``can_transform(source_type)``, which says whether the transform accepts a source
  column of an Arrow type. Iceberg uses the same name.

The generic code in ``cavebear.partition_spec``, ``cavebear.table`` and the engine
adapters uses only these three.
"""

import struct
import warnings
from dataclasses import dataclass
from typing import Any

import mmh3
import numpy as np
import pyarrow as pa

MAX_N_BUCKETS = 2**30
"""The largest bucket count of ``IcebergBucket``."""


def _check_members(data: dict[str, Any], allowed: set[str]) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"unknown members {sorted(extra)} for {data.get('@type')!r}")


# MurmurHash3 x86_32 constants.
_C1 = np.uint32(0xCC9E2D51)
_C2 = np.uint32(0x1B873593)
_M = np.uint32(5)
_N = np.uint32(0xE6546B64)
_F1 = np.uint32(0x85EBCA6B)
_F2 = np.uint32(0xC2B2AE35)


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def murmur3_x86_32_int64(keys: np.ndarray) -> np.ndarray:
    """MurmurHash3 x86_32 with seed 0 of the 8 little-endian bytes of each key.

    numpy ``uint32`` arithmetic wraps modulo 2**32, as murmur3 requires.

    Parameters
    ----------
    keys
        An ``int64`` or ``uint64`` array.

    Returns
    -------
    hashes :
        The unsigned 32-bit hash of each key, as ``uint32``.
    """
    keys = np.ascontiguousarray(keys).view(np.uint64)
    h = np.zeros(keys.shape, dtype=np.uint32)
    # An 8-byte key is two 4-byte blocks: the low 32 bits, then the high 32 bits.
    for block in (keys & np.uint64(0xFFFFFFFF), keys >> np.uint64(32)):
        k = block.astype(np.uint32)
        k = _rotl(k * _C1, 15) * _C2
        h = _rotl(h ^ k, 13) * _M + _N
    h ^= np.uint32(8)
    h ^= h >> np.uint32(16)
    h *= _F1
    h ^= h >> np.uint32(13)
    h *= _F2
    h ^= h >> np.uint32(16)
    return h


def _as_int64(values: pa.Array) -> pa.Array:
    """``values`` as ``int64``: by value, except that ``uint64`` is reinterpreted."""
    kind = values.type
    if not pa.types.is_integer(kind):
        raise TypeError(f"bucket transforms need an integer column, got {kind}")
    if kind == pa.uint64():
        return values.view(pa.int64())
    return values.cast(pa.int64())


def to_arrow_int64(values: Any) -> pa.Array:
    """Python integers, or an integer Arrow array, as an ``int64`` Arrow array.

    Python integers in ``[2**63, 2**64)`` are ``uint64`` values, and become the
    ``int64`` with the same 8 bytes. A ``uint64`` Arrow array is reinterpreted the
    same way.

    Parameters
    ----------
    values
        An iterable of Python integers or ``None``, or an integer Arrow array.

    Returns
    -------
    array :
        The values as ``int64``. ``None`` becomes null.

    Raises
    ------
    ValueError
        If a Python integer is outside the ``int64`` and ``uint64`` ranges.
    TypeError
        If an Arrow array does not have an integer type.
    """
    if isinstance(values, pa.Array):
        return _as_int64(values)
    converted = []
    for value in values:
        if value is not None:
            if not -(2**63) <= value < 2**64:
                raise ValueError(f"value {value} is outside the int64 and uint64 ranges")
            if value >= 2**63:
                value -= 2**64
        converted.append(value)
    return pa.array(converted, pa.int64())


@dataclass(frozen=True)
class IcebergBucket:
    """Apache Iceberg's ``bucket[N]`` transform, with N a power of two.

    See ``docs/specs/iceberg_bucket.md``.

    Parameters
    ----------
    n_buckets
        The bucket count N, a power of two in ``[1, 2**30]``.

    Raises
    ------
    ValueError
        If ``n_buckets`` is not a power of two in ``[1, 2**30]``.
    """

    n_buckets: int

    TYPE = "iceberg_bucket"
    """The ``"@type"`` of the transform object."""

    PARTITION_TYPE = pa.int32()
    """The Arrow type of the partition column (``docs/specs/iceberg_bucket.md``)."""

    def __post_init__(self):
        n = self.n_buckets
        if type(n) is not int or n < 1 or n > MAX_N_BUCKETS or n & (n - 1):
            raise ValueError(f"n_buckets must be a power of two in [1, 2**30], got {n!r}")

    @staticmethod
    def can_transform(source_type: pa.DataType) -> bool:
        """Whether the transform accepts a source column of ``source_type``.

        Parameters
        ----------
        source_type
            The Arrow type of the source column.

        Returns
        -------
        accepts :
            ``True`` for an integer type.
        """
        return pa.types.is_integer(source_type)

    def bucket(self, value: int) -> int:
        """The bucket number for an integer source value.

        This is the readable reference for ``apply``.

        Parameters
        ----------
        value
            The source value. Values in ``[2**63, 2**64)`` are uint64 values, and hash
            as the int64 with the same 8 bytes.

        Returns
        -------
        bucket :
            The bucket number, in ``[0, n_buckets)``.

        Raises
        ------
        ValueError
            If ``value`` is outside the int64 and uint64 ranges.
        """
        if not -(2**63) <= value < 2**64:
            raise ValueError(f"value {value} is outside the int64 and uint64 ranges")
        key = struct.pack("<Q" if value >= 2**63 else "<q", value)
        return (mmh3.hash(key, seed=0) & 0x7FFFFFFF) & (self.n_buckets - 1)

    def apply(self, values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
        """The bucket numbers of an integer Arrow array, as ``PARTITION_TYPE``.

        Parameters
        ----------
        values
            The source values. A ``uint64`` array hashes as its ``int64`` view.

        Returns
        -------
        buckets :
            The bucket numbers. A null value gives a null bucket. A chunked array
            gives a chunked array.

        Raises
        ------
        TypeError
            If ``values`` does not have an integer type.
        """
        if isinstance(values, pa.ChunkedArray):
            chunks = [self.apply(chunk) for chunk in values.chunks]
            return pa.chunked_array(chunks, self.PARTITION_TYPE)
        keys = _as_int64(values)
        hashes = murmur3_x86_32_int64(keys.fill_null(0).to_numpy(zero_copy_only=False))
        buckets = (hashes & np.uint32(self.n_buckets - 1)).astype(np.int32)
        mask = keys.is_null().to_numpy(zero_copy_only=False) if keys.null_count else None
        return pa.array(buckets, self.PARTITION_TYPE, mask=mask)

    def to_dict(self) -> dict[str, Any]:
        """The transform as a transform object (``docs/specs/iceberg_bucket.md``)."""
        return {"@type": self.TYPE, "n_buckets": self.n_buckets}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IcebergBucket":
        """The transform that a transform object describes.

        Parameters
        ----------
        data
            A transform object, as from ``to_dict``.

        Raises
        ------
        ValueError
            If ``data`` has unknown members, or a bad ``n_buckets``.
        """
        _check_members(data, {"@type", "n_buckets"})
        return cls(n_buckets=data["n_buckets"])


@dataclass(frozen=True)
class UnknownTransform:
    """A transform whose ``"@type"`` this version of cavebear does not know.

    Readers keep it, so that the partition spec round-trips, but never use it for
    pruning. Writers refuse a table that has one.

    Parameters
    ----------
    raw
        The transform object, as stored in the spec.
    """

    raw: dict[str, Any]

    @property
    def TYPE(self) -> Any:
        """The ``"@type"`` of the transform object."""
        return self.raw.get("@type")

    def to_dict(self) -> dict[str, Any]:
        """A copy of the transform object."""
        return dict(self.raw)


Transform = IcebergBucket | UnknownTransform
"""A transform of a partition field."""

TRANSFORMS: dict[str, type[IcebergBucket]] = {IcebergBucket.TYPE: IcebergBucket}
"""The known transforms, by ``"@type"``."""


def parse_transform(data: dict[str, Any]) -> Transform:
    """The transform that a partition spec's transform object describes.

    Parameters
    ----------
    data
        A transform object.

    Returns
    -------
    transform :
        The known transform, or an ``UnknownTransform`` with a warning if cavebear
        does not know the ``"@type"``.
    """
    kind = data.get("@type")
    if kind not in TRANSFORMS:
        warnings.warn(
            f"unknown transform @type {kind!r}: cavebear does not use it for pruning, "
            "so reads that filter on its source column open every file. "
            "A newer cavebear can know this transform.",
            stacklevel=2,
        )
        return UnknownTransform(dict(data))
    return TRANSFORMS[kind].from_dict(data)
