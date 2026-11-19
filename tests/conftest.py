import numpy as np
import pyarrow as pa
import pytest
from deltalake import DeltaTable

import cavebear
from cavebear import IcebergBucket, PartitionField, PartitionSpec

SCHEMA = pa.schema([("root_id", pa.int64()), ("value", pa.float64())])


def spec(n_buckets: int) -> PartitionSpec:
    return PartitionSpec((PartitionField("root_id_bucket", "root_id", IcebergBucket(n_buckets)),))


def synapses(n_roots=200, per_root=3, seed=0) -> pa.Table:
    """``per_root`` rows for each of ``n_roots`` random root IDs."""
    rng = np.random.default_rng(seed)
    roots = rng.integers(0, 2**62, size=n_roots, dtype=np.int64)
    return pa.table(
        {"root_id": np.repeat(roots, per_root), "value": rng.random(n_roots * per_root)},
        schema=SCHEMA,
    )


@pytest.fixture
def make_table(tmp_path):
    """A function that creates a table with ``n_buckets`` and writes ``data`` to it."""

    def make(name: str, n_buckets: int, data=None) -> DeltaTable:
        table = cavebear.create_delta(tmp_path / name, SCHEMA, spec(n_buckets))
        if data is not None:
            cavebear.write_delta(table, data)
        return table

    return make
