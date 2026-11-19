"""Synthetic tables and helpers that the Polars and DuckDB demos share."""

import numpy as np
import pyarrow as pa
from deltalake import DeltaTable

import cavebear
from cavebear import IcebergBucket, PartitionField, PartitionSpec

N_ROOTS = 2_000
N_SYNAPSES = 200_000

SYNAPSE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("pre_pt_root_id", pa.int64()),
        ("post_pt_root_id", pa.int64()),
        ("size", pa.int32()),
    ]
)
CELL_SCHEMA = pa.schema([("pt_root_id", pa.int64()), ("cell_type", pa.string())])

#: 64 partitions on the postsynaptic root ID.
SYNAPSE_SPEC = PartitionSpec(
    (PartitionField("post_pt_root_id_bucket", "post_pt_root_id", IcebergBucket(64)),)
)
#: 16 partitions on the root ID. bucket[64] is nested in bucket[16].
CELL_SPEC = PartitionSpec((PartitionField("pt_root_id_bucket", "pt_root_id", IcebergBucket(16)),))


def synthetic_tables(seed: int = 0) -> tuple[pa.Table, pa.Table]:
    """Random synapses between ``N_ROOTS`` root IDs, and a cell type for each root."""
    rng = np.random.default_rng(seed)
    # Unique IDs in the range of real root IDs, so each root has one cell type.
    roots = np.unique(rng.integers(864691135_000_000_000, 864691137_000_000_000, N_ROOTS))
    synapses = pa.table(
        {
            "id": np.arange(N_SYNAPSES),
            "pre_pt_root_id": rng.choice(roots, N_SYNAPSES),
            "post_pt_root_id": rng.choice(roots, N_SYNAPSES),
            "size": rng.integers(10, 5_000, N_SYNAPSES, dtype=np.int32),
        },
        schema=SYNAPSE_SCHEMA,
    )
    cells = pa.table(
        {"pt_root_id": roots, "cell_type": rng.choice(["excitatory", "inhibitory"], len(roots))},
        schema=CELL_SCHEMA,
    )
    return synapses, cells


def files_opened(dt: DeltaTable, source: str, roots: list[int]) -> str:
    """The files that a read of ``roots`` opens, with and without the partition filter.

    Without the partition filter, the only filter is on the source column. Delta can
    then skip a file only by its min and max values of that column.
    """
    total = len(dt.file_uris())
    plain = len(dt.file_uris(file_pruning_predicate=[(source, "in", roots)]))
    values = cavebear.partition_values(dt, {source: roots})
    pruned = len(dt.file_uris(file_pruning_predicate=[(n, "in", v) for n, v in values.items()]))
    return f"{pruned} of {total} files (without the partition filter: {plain} of {total})"


def sorted_rows(table: pa.Table, columns: list[str]) -> list[tuple]:
    table = table.select(columns)
    return sorted(zip(*(column.to_pylist() for column in table.columns)))
