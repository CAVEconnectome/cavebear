import numpy as np
import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake

import cavebear
from cavebear import IcebergBucket, PartitionField, PartitionSpec, nested_partitions

SPEC = PartitionSpec((PartitionField("root_id_bucket", "root_id", IcebergBucket(8)),))
SCHEMA = pa.schema([("root_id", pa.int64()), ("value", pa.float64())])


def make_data(n=1000, seed=0) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table(
        {"root_id": rng.integers(0, 2**62, size=n, dtype=np.int64), "value": rng.random(n)},
        schema=SCHEMA,
    )


@pytest.fixture
def table(tmp_path) -> DeltaTable:
    return cavebear.create_delta(tmp_path / "table", SCHEMA, SPEC)


def test_create(table):
    assert cavebear.partition_spec(table) == SPEC
    assert table.version() == 0
    assert table.metadata().partition_columns == ["root_id_bucket"]
    assert pa.schema(table.schema().to_arrow()).field("root_id_bucket").type == pa.int32()


def test_write_computes_buckets(table):
    data = make_data()
    assert cavebear.write_delta(table, data) is None
    # As with write_deltalake, the DeltaTable moves to the new version.
    assert table.version() == 1
    got = table.to_pyarrow_table().sort_by("root_id")
    expected = data.sort_by("root_id")
    assert got.column("root_id") == expected.column("root_id")
    assert got.column("root_id_bucket").to_pylist() == [
        IcebergBucket(8).bucket(v) for v in expected.column("root_id").to_pylist()
    ]
    # One directory per bucket.
    directories = {uri.split("/")[-2] for uri in table.file_uris()}
    assert directories == {f"root_id_bucket={b}" for b in range(8)}


def test_write_by_uri_and_overwrite(table):
    cavebear.write_delta(table.table_uri, make_data(10))
    assert cavebear.write_delta(table.table_uri, make_data(5, seed=1), mode="overwrite") is None
    new = DeltaTable(table.table_uri)
    assert new.to_pyarrow_table().num_rows == 5
    assert cavebear.partition_spec(new) == SPEC
    # The DeltaTable that was not passed stays at its version.
    assert table.version() == 0


def test_write_rejects_bucket_columns(table):
    data = make_data(3).append_column("root_id_bucket", pa.array([0, 0, 0], pa.int32()))
    with pytest.raises(ValueError, match="already has the partition column"):
        cavebear.write_delta(table, data)


def test_create_rejects_bad_schemas(tmp_path):
    with pytest.raises(ValueError, match="not in the schema"):
        cavebear.create_delta(tmp_path / "a", pa.schema([("value", pa.float64())]), SPEC)
    with pytest.raises(ValueError, match="does not accept"):
        cavebear.create_delta(tmp_path / "f", pa.schema([("root_id", pa.float64())]), SPEC)
    with pytest.raises(ValueError, match="already has the partition column"):
        cavebear.create_delta(tmp_path / "b", SCHEMA.append(pa.field("root_id_bucket", pa.int32())), SPEC)
    with pytest.raises(ValueError, match="not a partition column"):
        cavebear.create_delta(tmp_path / "c", SCHEMA, SPEC, partition_by=["value"])


def test_open_rejects_plain_and_stale_tables(tmp_path, table):
    plain = tmp_path / "plain"
    write_deltalake(plain, make_data(3))
    with pytest.raises(ValueError, match="no cavebear partition spec"):
        cavebear.partition_spec(DeltaTable(plain))
    # An overwrite that changes the partition columns keeps the spec key.
    write_deltalake(
        table, make_data(3), mode="overwrite", schema_mode="overwrite", partition_by=["root_id"]
    )
    with pytest.raises(ValueError, match="not a partition column"):
        cavebear.partition_spec(table)


def test_spec_of_an_old_version(table):
    cavebear.write_delta(table, make_data(10))
    old = DeltaTable(table.table_uri, version=0)
    assert cavebear.partition_spec(old) == SPEC
    assert old.to_pyarrow_table().num_rows == 0


def test_partition_values(tmp_path):
    # The test vectors of docs/specs/iceberg_bucket.md, with n_buckets = 16.
    spec = PartitionSpec((PartitionField("b", "root_id", IcebergBucket(16)),))
    table = cavebear.create_delta(tmp_path / "sixteen", SCHEMA, spec)
    values = cavebear.partition_values
    assert values(table, {"root_id": [34, 0, 2**64 - 34]}) == {"b": [3, 12, 13]}
    assert values(table, {"root_id": [34, None]}) == {"b": [3]}
    assert values(table, {"value": [1]}) == {}


def test_nested_partitions():
    assert nested_partitions(5, 64, 128) == [5, 69]
    assert nested_partitions(3, 4, 16) == [3, 7, 11, 15]
    assert nested_partitions(0, 1, 2) == [0, 1]
    with pytest.raises(ValueError):
        nested_partitions(0, 8, 4)
    with pytest.raises(ValueError):
        nested_partitions(4, 4, 16)
    with pytest.raises(ValueError):
        nested_partitions(0, 3, 12)
