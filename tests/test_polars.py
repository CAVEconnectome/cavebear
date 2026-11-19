import re

import pytest

pl = pytest.importorskip("polars")

import cavebear  # noqa: E402
from cavebear import IcebergBucket  # noqa: E402
from cavebear import polars as cpl  # noqa: E402

from conftest import synapses  # noqa: E402


def test_predicate_matches_plain_filter_and_prunes(make_table):
    data = synapses()
    table = make_table("t", 16, data)
    roots = data.column("root_id").to_pylist()[::150]
    lf = pl.scan_delta(table).filter(cpl.predicate(table, {"root_id": roots}))
    expected = pl.DataFrame(data).filter(pl.col("root_id").is_in(roots))
    got = lf.collect().drop("root_id_bucket")
    assert got.sort("root_id", "value").equals(expected.sort("root_id", "value"))
    # The plan lists only the files of the buckets of the roots.
    n_buckets = len(cavebear.partition_values(table, {"root_id": roots})["root_id_bucket"])
    assert files_scanned(lf) == n_buckets < 16


def files_scanned(lf) -> int:
    """The number of files in the scan of a plan. Polars lists the first file and
    shortens the rest to "... n other sources"."""
    scan = lf.explain().splitlines()[0]
    other = re.search(r"\.\.\. (\d+) other sources", scan)
    return scan.count(".parquet") + (int(other.group(1)) if other else 0)


def test_partition_expr():
    df = pl.DataFrame({"x": pl.Series([34, 0, None, -34], dtype=pl.Int64)})
    got = df.select(cpl.partition_expr("x", IcebergBucket(16))).to_series()
    assert got.dtype == pl.Int32
    assert got.to_list() == [3, 12, None, 13]
    unsigned = df.select(pl.col("x").reinterpret(signed=False))
    assert unsigned.select(cpl.partition_expr("x", IcebergBucket(16))).to_series().to_list() == [3, 12, None, 13]


def test_write_from_polars(make_table):
    table = make_table("t", 8)
    table = cavebear.write(table, pl.DataFrame(synapses(20)))
    assert pl.scan_delta(table).collect().height == 60


def test_scan_partitionwise(make_table):
    # bucket[128] is nested in bucket[64]: partition v of 64 holds v and v + 64 of 128.
    data = synapses(300, per_root=3, seed=1)
    left = make_table("left", 64, data)
    right = make_table("right", 128, data)
    full = pl.scan_delta(left).join(pl.scan_delta(right), on="root_id").collect()
    parts = []
    for value, (lf_left, lf_right) in cpl.scan_partitionwise([left, right], on="root_id"):
        assert value < 64
        parts.append(lf_left.join(lf_right, on="root_id").collect())
    chunked = pl.concat(parts)
    columns = ["root_id", "value", "value_right"]
    # Each root has 3 rows on each side, so 3 * 3 joined rows.
    assert chunked.height == full.height == 300 * 3 * 3
    assert chunked.select(columns).sort(columns).equals(full.select(columns).sort(columns))


def test_scan_partitionwise_rejects_tables_without_fields(make_table):
    table = make_table("t", 8)
    with pytest.raises(ValueError, match="no iceberg_bucket field"):
        next(cpl.scan_partitionwise([table], on="value", n=4))


def test_native_write(make_table):
    from deltalake import write_deltalake

    data = synapses()
    table = make_table("t", 16)
    df = pl.DataFrame(data)
    write_deltalake(table, df.with_columns(cpl.partition_columns(table)), mode="append")
    # The same partition columns as the shortcut write.
    expected_table = make_table("expected", 16, data)
    columns = ["root_id", "value", "root_id_bucket"]
    got_all = pl.scan_delta(table).collect().select(columns).sort(columns)
    assert got_all.equals(pl.scan_delta(expected_table).collect().select(columns).sort(columns))


def test_expressions_take_a_spec():
    from conftest import spec

    df = pl.DataFrame({"root_id": pl.Series([34, 0, None], dtype=pl.Int64)})
    df = df.with_columns(cpl.partition_columns(spec(16)))
    assert df["root_id_bucket"].to_list() == [3, 12, None]
    assert df.filter(cpl.predicate(spec(16), {"root_id": [34]})).height == 1
    assert df.filter(cpl.predicate(spec(16), {})).height == 3
