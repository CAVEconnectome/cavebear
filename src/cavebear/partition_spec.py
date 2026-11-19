"""The partition spec of a cavebear table: which partition columns hold which transforms.

See ``docs/specs/partition_spec.md``. The spec is stored as a JSON string under
``CONFIG_KEY`` in the Delta table ``configuration``.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .transforms import Transform, UnknownTransform, parse_transform

CONFIG_KEY = "cave.partition_spec"
SPEC_TYPE = "cave_partition_spec_v1"


@dataclass(frozen=True)
class PartitionField:
    """One Delta partition column, computed by ``transform`` from ``source``."""

    name: str
    source: str
    transform: Transform

    @property
    def known(self) -> bool:
        """Whether this version of cavebear can compute and prune on this field."""
        return not isinstance(self.transform, UnknownTransform)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "transform": self.transform.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionField":
        extra = set(data) - {"name", "source", "transform"}
        if extra:
            raise ValueError(f"unknown partition field members {sorted(extra)}")
        return cls(
            name=data["name"],
            source=data["source"],
            transform=parse_transform(data["transform"]),
        )


@dataclass(frozen=True)
class PartitionSpec:
    """The partition fields of one table."""

    fields: tuple[PartitionField, ...]

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields:
            raise ValueError("a partition spec needs at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"partition field names are not unique: {names}")

    def fields_for(self, source: str) -> list[PartitionField]:
        """The known partition fields that are computed from the column ``source``."""
        return [f for f in self.fields if f.source == source and f.known]

    def require_known(self) -> None:
        """Raise ``ValueError`` if a field has a transform that cavebear does not know.

        Writers call this, because they cannot compute such a field.
        """
        unknown = [f"{f.name} ({f.transform.TYPE!r})" for f in self.fields if not f.known]
        if unknown:
            raise ValueError(f"cannot write fields with unknown transforms: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        return {"@type": SPEC_TYPE, "fields": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionSpec":
        if data.get("@type") != SPEC_TYPE:
            raise ValueError(f"partition spec @type must be {SPEC_TYPE!r}, got {data.get('@type')!r}")
        extra = set(data) - {"@type", "fields"}
        if extra:
            raise ValueError(f"unknown partition spec members {sorted(extra)}")
        fields = data.get("fields")
        if not isinstance(fields, list):
            raise ValueError("partition spec fields must be a list")
        return cls(fields=tuple(PartitionField.from_dict(field) for field in fields))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str | bytes) -> "PartitionSpec":
        return cls.from_dict(json.loads(text))

    def to_configuration(self) -> dict[str, str]:
        """The Delta table ``configuration`` entry that stores this spec."""
        return {CONFIG_KEY: self.to_json()}

    @classmethod
    def from_configuration(cls, configuration: Mapping[str, str]) -> "PartitionSpec | None":
        """The spec in a Delta table ``configuration``, or ``None`` without one."""
        text = configuration.get(CONFIG_KEY)
        return None if text is None else cls.from_json(text)
