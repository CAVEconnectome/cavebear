"""The partition spec of a cavebear table: which partition columns hold which transforms.

See ``docs/specs/partition_spec.md``. The spec is stored as a JSON string under
``CONFIG_KEY`` in the Delta table ``configuration``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

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

    def arrow_field(self) -> pa.Field:
        """The Arrow field of the partition column. The field must be known."""
        return pa.field(self.name, self.transform.PARTITION_TYPE)

    def check_source(self, schema: pa.Schema) -> None:
        """Raise ``ValueError`` if ``schema`` has no source column that the transform
        accepts. For an unknown transform, only check that the column exists."""
        if self.source not in schema.names:
            raise ValueError(f"source column {self.source!r} of {self.name!r} is not in the schema")
        source_type = schema.field(self.source).type
        if self.known and not self.transform.can_transform(source_type):
            raise ValueError(
                f"source column {self.source!r} has type {source_type}, "
                f"which {self.transform.TYPE!r} does not accept"
            )

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

    def add_partition_columns(self, schema: pa.Schema) -> pa.Schema:
        """``schema`` with the partition columns of the spec added at the end.

        ``schema`` is the schema of data to write, which has the source columns but
        not the partition columns, because cavebear computes them. Raises
        ``ValueError`` if a transform is unknown, if ``schema`` already has a partition
        column, or if a transform does not accept its source column.
        """
        self.require_known()
        for f in self.fields:
            if f.name in schema.names:
                raise ValueError(f"the data already has the partition column {f.name!r}; cavebear computes it")
            f.check_source(schema)
        return pa.schema([*schema, *(f.arrow_field() for f in self.fields)])

    def validate_table(self, schema: pa.Schema, partition_columns: Sequence[str]) -> None:
        """Raise ``ValueError`` if the spec does not describe a table.

        ``schema`` is the full schema of the table, and ``partition_columns`` its Delta
        partition columns. Each field must name a partition column, and its source
        must be a column of the table. For a known transform, the partition column
        must also have the type that the transform defines, and the transform must
        accept the source column. A table fails this check after an overwrite that
        changes its partition columns, because the spec key survives it.
        """
        for f in self.fields:
            if f.name not in partition_columns:
                raise ValueError(f"partition spec field {f.name!r} is not a partition column")
            f.check_source(schema)
            if f.known and schema.field(f.name).type != f.transform.PARTITION_TYPE:
                raise ValueError(f"partition column {f.name!r} is not {f.transform.PARTITION_TYPE}")

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
