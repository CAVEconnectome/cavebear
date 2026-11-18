"""Partition transforms: deterministic maps from a source value to a bucket number.

Each transform is a frozen dataclass whose ``to_dict`` gives the transform object
of a layout (``docs/specs/layout.md``), with an ``"@type"`` member that names it.
"""

import struct
from dataclasses import dataclass
from typing import Any

import mmh3

MAX_N_BUCKETS = 2**30


def _check_members(data: dict[str, Any], allowed: set[str]) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"unknown members {sorted(extra)} for {data.get('@type')!r}")


@dataclass(frozen=True)
class IcebergBucket:
    """Apache Iceberg's ``bucket[N]`` transform, with N a power of two.

    See ``docs/specs/iceberg_bucket.md``.
    """

    n_buckets: int

    TYPE = "iceberg_bucket"

    def __post_init__(self):
        n = self.n_buckets
        if type(n) is not int or n < 1 or n > MAX_N_BUCKETS or n & (n - 1):
            raise ValueError(f"n_buckets must be a power of two in [1, 2**30], got {n!r}")

    def bucket(self, value: int) -> int:
        """The bucket number for an integer source value.

        Values in ``[2**63, 2**64)`` are uint64 values, and hash as the int64 with
        the same 8 bytes.
        """
        if not -(2**63) <= value < 2**64:
            raise ValueError(f"value {value} is outside the int64 and uint64 ranges")
        key = struct.pack("<Q" if value >= 2**63 else "<q", value)
        return (mmh3.hash(key, seed=0) & 0x7FFFFFFF) & (self.n_buckets - 1)

    def to_dict(self) -> dict[str, Any]:
        return {"@type": self.TYPE, "n_buckets": self.n_buckets}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IcebergBucket":
        _check_members(data, {"@type", "n_buckets"})
        return cls(n_buckets=data["n_buckets"])


Transform = IcebergBucket

TRANSFORMS: dict[str, type[Transform]] = {IcebergBucket.TYPE: IcebergBucket}


def parse_transform(data: dict[str, Any]) -> Transform:
    """The transform that a layout's transform object describes."""
    kind = data.get("@type")
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown transform @type {kind!r}")
    return TRANSFORMS[kind].from_dict(data)
