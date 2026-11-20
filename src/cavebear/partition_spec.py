"""The partition spec of a Delta table: which partition columns hold which transforms.

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
"""The key of the partition spec in the Delta table ``configuration``."""

SPEC_TYPE = "cave_partition_spec_v1"
"""The ``"@type"`` of a partition spec object."""


DEFAULT_NAME_SUFFIX = "_partition"
"""The suffix of the default partition column name: ``f"{source}_partition"``."""


@dataclass(frozen=True, kw_only=True)
class PartitionField:
    """One Delta partition column, computed by ``transform`` from ``source``.

    All arguments are keyword-only.

    Parameters
    ----------
    source
        The name of the source column.
    transform
        The transform that maps a source value to a partition value.
    name
        The name of the partition column. The default is ``f"{source}_partition"``.

    Examples
    --------
    >>> from cavebear import IcebergBucket
    >>> PartitionField(source="pt_root_id", transform=IcebergBucket(64)).name
    'pt_root_id_partition'
    """

    source: str
    transform: Transform
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            object.__setattr__(self, "name", f"{self.source}{DEFAULT_NAME_SUFFIX}")

    @property
    def known(self) -> bool:
        """Whether this version of cavebear can compute and prune on this field."""
        return not isinstance(self.transform, UnknownTransform)

    def arrow_field(self) -> pa.Field:
        """The Arrow field of the partition column. The field must be known.

        Returns
        -------
        field :
            The field, with the name of the partition column and the
            ``PARTITION_TYPE`` of the transform.
        """
        return pa.field(self.name, self.transform.PARTITION_TYPE)

    def check_source(self, schema: pa.Schema) -> None:
        """Check that ``schema`` has a source column that the transform accepts.

        For an unknown transform, only check that the column exists.

        Parameters
        ----------
        schema
            The schema to check.

        Raises
        ------
        ValueError
            If the source column is missing, or if the transform does not accept its
            type.
        """
        if self.source not in schema.names:
            raise ValueError(f"source column {self.source!r} of {self.name!r} is not in the schema")
        source_type = schema.field(self.source).type
        if self.known and not self.transform.can_transform(source_type):
            raise ValueError(
                f"source column {self.source!r} has type {source_type}, "
                f"which {self.transform.TYPE!r} does not accept"
            )

    def to_dict(self) -> dict[str, Any]:
        """The field as a partition field object (``docs/specs/partition_spec.md``)."""
        return {"name": self.name, "source": self.source, "transform": self.transform.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionField":
        """The field that a partition field object describes.

        Parameters
        ----------
        data
            A partition field object, as from ``to_dict``.

        Raises
        ------
        ValueError
            If ``data`` has unknown members.
        """
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
    """The partition fields of one table.

    Parameters
    ----------
    fields
        The partition fields, as one of:

        - A mapping from source column to transform. Each entry becomes a field with
          the default name (``f"{source}_partition"``), in mapping order.
        - A sequence of ``PartitionField``, for a custom name or for two fields on
          one source column.

        Either one becomes a tuple of ``PartitionField``.

    Raises
    ------
    ValueError
        If ``fields`` is empty, or if two fields have the same name.

    Examples
    --------
    One field, with the default name:

    >>> from cavebear import IcebergBucket
    >>> spec = PartitionSpec({"pt_root_id": IcebergBucket(64)})
    >>> [f.name for f in spec.fields]
    ['pt_root_id_partition']

    Two fields on one source column need the full form, with a name for at least one
    of them:

    >>> spec = PartitionSpec((
    ...     PartitionField(source="pt_root_id", transform=IcebergBucket(64)),
    ...     PartitionField(source="pt_root_id", transform=IcebergBucket(1024), name="pt_root_id_1024"),
    ... ))
    >>> [f.name for f in spec.fields]
    ['pt_root_id_partition', 'pt_root_id_1024']
    """

    fields: tuple[PartitionField, ...]

    def __post_init__(self):
        fields = self.fields
        if isinstance(fields, Mapping):
            fields = (PartitionField(source=source, transform=t) for source, t in fields.items())
        object.__setattr__(self, "fields", tuple(fields))
        if not self.fields:
            raise ValueError("a partition spec needs at least one field")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"partition field names are not unique: {names}")

    def fields_for(self, source: str) -> list[PartitionField]:
        """The known partition fields that are computed from a source column.

        Parameters
        ----------
        source
            The name of the source column.

        Returns
        -------
        fields :
            The known fields on ``source``, in spec order.
        """
        return [f for f in self.fields if f.source == source and f.known]

    def require_known(self) -> None:
        """Check that cavebear knows the transform of each field.

        Writers call this, because they cannot compute a field with an unknown
        transform.

        Raises
        ------
        ValueError
            If a field has a transform that cavebear does not know.
        """
        unknown = [f"{f.name} ({f.transform.TYPE!r})" for f in self.fields if not f.known]
        if unknown:
            raise ValueError(f"cannot write fields with unknown transforms: {unknown}")

    def add_partition_columns(self, schema: pa.Schema) -> pa.Schema:
        """``schema`` with the partition columns of the spec added at the end.

        Parameters
        ----------
        schema
            The schema of data to write. It has the source columns but not the
            partition columns, because cavebear computes them.

        Returns
        -------
        schema :
            The schema with the partition columns.

        Raises
        ------
        ValueError
            If a transform is unknown, if ``schema`` already has a partition column,
            or if a transform does not accept its source column.
        """
        self.require_known()
        for f in self.fields:
            if f.name in schema.names:
                raise ValueError(f"the data already has the partition column {f.name!r}; cavebear computes it")
            f.check_source(schema)
        return pa.schema([*schema, *(f.arrow_field() for f in self.fields)])

    def partition_arrays(self, data: pa.RecordBatch | pa.Table) -> dict[str, pa.Array | pa.ChunkedArray]:
        """The partition columns of the spec, computed from the source columns of ``data``.

        Use it to add the partition columns to data that you write with your own
        ``write_deltalake`` calls. ``data`` is not copied.

        Parameters
        ----------
        data
            A record batch or table that has the source columns but not the
            partition columns.

        Returns
        -------
        arrays :
            Maps each partition column name to its values, in spec order. The values
            are an ``Array`` for a record batch and a ``ChunkedArray`` for a table.

        Raises
        ------
        ValueError
            As ``add_partition_columns``, for the schema of ``data``.
        """
        self.add_partition_columns(data.schema)
        return {f.name: f.transform.apply(data.column(f.source)) for f in self.fields}

    def validate_table(self, schema: pa.Schema, partition_columns: Sequence[str]) -> None:
        """Check that the spec describes a table.

        Each field must name a partition column, and its source must be a column of
        the table. For a known transform, the partition column must also have the
        type that the transform defines, and the transform must accept the source
        column. A table fails this check after an overwrite that changes its
        partition columns, because the spec key survives it.

        Parameters
        ----------
        schema
            The full schema of the table.
        partition_columns
            The Delta partition columns of the table.

        Raises
        ------
        ValueError
            If the spec does not describe the table.
        """
        for f in self.fields:
            if f.name not in partition_columns:
                raise ValueError(f"partition spec field {f.name!r} is not a partition column")
            f.check_source(schema)
            if f.known and schema.field(f.name).type != f.transform.PARTITION_TYPE:
                raise ValueError(f"partition column {f.name!r} is not {f.transform.PARTITION_TYPE}")

    def to_dict(self) -> dict[str, Any]:
        """The spec as a partition spec object (``docs/specs/partition_spec.md``)."""
        return {"@type": SPEC_TYPE, "fields": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionSpec":
        """The spec that a partition spec object describes.

        Parameters
        ----------
        data
            A partition spec object, as from ``to_dict``.

        Raises
        ------
        ValueError
            If ``data`` does not have the ``"@type"`` ``SPEC_TYPE``, has unknown
            members, or has no list of fields.
        """
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
        """The spec as compact JSON text."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str | bytes) -> "PartitionSpec":
        """The spec in JSON text, as from ``to_json``.

        Parameters
        ----------
        text
            The JSON text of a partition spec object.
        """
        return cls.from_dict(json.loads(text))

    def to_configuration(self) -> dict[str, str]:
        """The Delta table ``configuration`` entry that stores this spec."""
        return {CONFIG_KEY: self.to_json()}

    @classmethod
    def from_configuration(cls, configuration: Mapping[str, str]) -> "PartitionSpec | None":
        """The spec in a Delta table ``configuration``.

        Parameters
        ----------
        configuration
            The ``configuration`` of a Delta table.

        Returns
        -------
        spec :
            The spec under ``CONFIG_KEY``, or ``None`` if the key is not set.
        """
        text = configuration.get(CONFIG_KEY)
        return None if text is None else cls.from_json(text)
