"""The partition spec as a custom key in the Delta table configuration.

These tests are the evidence for that decision (see plans/decisions.md). They check
that each operation that cavebear users run keeps the key, and that no table feature
is necessary for it.
"""

import json

import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError

KEY = "cave.partition_spec"
VALUE = json.dumps(
    {
        "@type": "cave_partition_spec_v1",
        "fields": [
            {
                "name": "id_partition",
                "source": "id",
                "transform": {"@type": "iceberg_bucket", "n_buckets": 4},
            }
        ],
    }
)
DATA = pa.table(
    {
        "id": pa.array([1, 2, 3, 4], pa.int64()),
        "value": pa.array([1.0, 2.0, 3.0, 4.0]),
        "id_partition": pa.array([0, 1, 0, 3], pa.int32()),
    }
)


def has_key(uri, **kwargs) -> bool:
    return DeltaTable(uri, **kwargs).metadata().configuration.get(KEY) == VALUE


@pytest.fixture
def table(tmp_path) -> str:
    uri = str(tmp_path / "table")
    DeltaTable.create(
        uri,
        DATA.schema,
        partition_by=["id_partition"],
        configuration={KEY: VALUE},
        raise_if_key_not_exists=False,
    )
    write_deltalake(uri, DATA, mode="append")
    return uri


def test_custom_key_needs_the_flag(tmp_path):
    with pytest.raises(DeltaError, match="Error parsing property"):
        DeltaTable.create(str(tmp_path / "a"), DATA.schema, configuration={KEY: VALUE})
    # write_deltalake has no flag, so it cannot create a table with the key.
    with pytest.raises(DeltaError, match="Error parsing property"):
        write_deltalake(str(tmp_path / "b"), DATA, configuration={KEY: VALUE})


def test_create_needs_no_table_feature(table):
    assert has_key(table)
    protocol = DeltaTable(table).protocol()
    assert (protocol.min_reader_version, protocol.min_writer_version) == (1, 2)
    assert protocol.reader_features is None and protocol.writer_features is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "append"},
        {"mode": "overwrite"},
        {"mode": "overwrite", "schema_mode": "overwrite"},
        {"mode": "append", "schema_mode": "merge"},
    ],
)
def test_writes_keep_key(table, kwargs):
    write_deltalake(table, DATA, **kwargs)
    assert has_key(table)


def test_maintenance_keeps_key(table):
    dt = DeltaTable(table)
    dt.optimize.compact()
    dt.optimize.z_order(["id"])
    dt.delete("id = 1")
    dt.merge(DATA, "s.id = t.id", source_alias="s", target_alias="t").when_matched_update_all().when_not_matched_insert_all().execute()
    dt.create_checkpoint()
    dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    assert has_key(table)
    assert has_key(table, version=0)


def test_key_survives_a_partition_change(table):
    # The key is not tied to the partition columns. A reader must check that each
    # field of the spec is still a partition column.
    write_deltalake(table, DATA, mode="overwrite", schema_mode="overwrite", partition_by=["id"])
    assert has_key(table)
    assert DeltaTable(table).metadata().partition_columns == ["id"]


def test_set_on_existing_table(tmp_path):
    uri = str(tmp_path / "table")
    write_deltalake(uri, DATA, partition_by=["id_partition"])
    with pytest.raises(DeltaError, match="Error parsing property"):
        DeltaTable(uri).alter.set_table_properties({KEY: VALUE})
    DeltaTable(uri).alter.set_table_properties({KEY: VALUE}, raise_if_not_exists=False)
    assert has_key(uri)


def test_polars_reads_and_writes_keep_key(table):
    pl = pytest.importorskip("polars")
    assert pl.scan_delta(table).collect().height == DATA.num_rows
    pl.DataFrame(DATA).write_delta(table, mode="append")
    pl.DataFrame(DATA).write_delta(table, mode="overwrite")
    assert has_key(table)


def test_duckdb_reads_and_writes_keep_key(table):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    assert con.sql(f"SELECT count(*) FROM delta_scan('{table}')").fetchall() == [(4,)]
    con.sql(f"ATTACH '{table}' AS latest (TYPE delta)")
    con.sql("INSERT INTO latest SELECT 5, 5.0, 2")
    assert has_key(table)
    # ATTACH at a version gives the same snapshot as DeltaTable(uri, version=...).
    con.sql(f"ATTACH '{table}' AS first (TYPE delta, VERSION 1)")
    assert con.sql("SELECT count(*) FROM first").fetchall() == [(4,)]
