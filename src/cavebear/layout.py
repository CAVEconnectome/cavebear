"""The layout of a cavebear table: which partition columns hold which transforms.

See ``docs/specs/layout.md``. The layout is stored as JSON at ``LAYOUT_PATH``,
relative to the Delta table root.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._store import open_store, read_bytes, write_bytes
from .transforms import Transform, parse_transform

LAYOUT_PATH = "_info/layout.json"
LAYOUT_TYPE = "cave_layout_v1"


@dataclass(frozen=True)
class PartitionField:
    """One Delta partition column, computed by ``transform`` from ``source``."""

    name: str
    source: str
    transform: Transform

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
class Layout:
    """The partition fields of one table."""

    fields: tuple[PartitionField, ...]

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"partition field names are not unique: {names}")

    def to_dict(self) -> dict[str, Any]:
        return {"@type": LAYOUT_TYPE, "transforms": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layout":
        if data.get("@type") != LAYOUT_TYPE:
            raise ValueError(f"layout @type must be {LAYOUT_TYPE!r}, got {data.get('@type')!r}")
        transforms = data.get("transforms")
        if not isinstance(transforms, list):
            raise ValueError("layout transforms must be a list")
        return cls(fields=tuple(PartitionField.from_dict(field) for field in transforms))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str | bytes) -> "Layout":
        return cls.from_dict(json.loads(text))

    def write(
        self,
        table_root: str | Path,
        storage_options: dict[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Write this layout to ``<table_root>/_info/layout.json``.

        Without ``overwrite``, an existing layout raises ``FileExistsError``,
        because a change of layout needs a rewrite of the table.
        """
        store = open_store(table_root, storage_options, mkdir=True)
        write_bytes(store, LAYOUT_PATH, self.to_json().encode(), overwrite=overwrite)

    @classmethod
    def read(cls, table_root: str | Path, storage_options: dict[str, Any] | None = None) -> "Layout":
        """The layout of the table at ``table_root``.

        A table without a layout raises ``FileNotFoundError``.
        """
        store = open_store(table_root, storage_options)
        return cls.from_json(read_bytes(store, LAYOUT_PATH))
