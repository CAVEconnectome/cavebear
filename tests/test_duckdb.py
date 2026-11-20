import re

import pyarrow as pa
import pytest
from deltalake import DeltaTable

duckdb = pytest.importorskip("duckdb")

import cavebear  # noqa: E402
from cavebear import IcebergBucket  # noqa: E402
from cavebear import duckdb as cdd  # noqa: E402

from conftest import synapses  # noqa: E402


def files_read(con, relation) -> int:
    profile = con.sql("EXPLAIN ANALYZE " + relation.sql_query()).fetchall()[0][1]
    return int(re.search(r"Total Files Read: (\d+)", profile).group(1))


def sorted_rows(relation_or_table) -> list[tuple]:
    if isinstance(relation_or_table, pa.Table):
        return sorted(zip(*(c.to_pylist() for c in relation_or_table.columns)))
    return sorted(relation_or_table.fetchall())


def delta_scan(con, table):
    return con.sql(f"FROM delta_scan('{table.table_uri}')")


def test_predicate_matches_plain_filter_and_prunes(make_table):
    con = duckdb.connect()
    data = synapses()
    table = make_table("t", 16, data)
    roots = data.column("root_id").to_pylist()[::150]
    relation = delta_scan(con, table).filter(cdd.predicate(table, {"root_id": roots}))
    got = relation.select("root_id, value")
    expected = data.filter(pa.compute.is_in(data.column("root_id"), pa.array(roots)))
    assert sorted_rows(got) == sorted_rows(expected)
    n_buckets = len(cavebear.partition_values(table, {"root_id": roots})["root_id_partition"])
    assert files_read(con, relation) == n_buckets < 16


def test_scan_delta_partitionwise_join_reads_the_table_version(make_table):
    con = duckdb.connect()
    table = make_table("t", 4, synapses(10))
    old = DeltaTable(table.table_uri, version=table.version())
    cavebear.write_delta(table, synapses(10, seed=1))
    # DuckDB reads the version of each DeltaTable, not the latest one.
    def rows(dt):
        return sum(len(r.fetchall()) for _, (r,) in cdd.scan_delta_partitionwise_join(con, [dt], on="root_id"))
    assert rows(old) == 30
    assert rows(table) == 60


def test_registered_function():
    con = duckdb.connect()
    cdd.register(con)
    rows = con.sql(
        "SELECT cave_iceberg_bucket(x, 16) FROM (VALUES (34), (0), (NULL), (-34)) t(x)"
    ).fetchall()
    assert [r[0] for r in rows] == [3, 12, None, 13]
    vectors = con.sql("SELECT cave_iceberg_bucket(x, 1024) FROM range(1000) t(x)").fetchall()
    assert [r[0] for r in vectors] == [IcebergBucket(1024).bucket(v) for v in range(1000)]


def test_write_from_duckdb(make_table):
    con = duckdb.connect()
    table = make_table("t", 8)
    relation = con.from_arrow(synapses(20))
    cavebear.write_delta(table, relation)
    assert len(delta_scan(con, table).fetchall()) == 60


def test_cross_engine(make_table):
    pl = pytest.importorskip("polars")
    from cavebear import polars as cpl

    con = duckdb.connect()
    data = synapses(50)
    roots = data.column("root_id").to_pylist()[:6]
    # Written with Polars, read with DuckDB.
    a = make_table("a", 16, pl.DataFrame(data))
    # Written with DuckDB, read with Polars.
    b = make_table("b", 16, con.from_arrow(data))
    from_duckdb = sorted_rows(delta_scan(con, a).filter(cdd.predicate(a, {"root_id": roots})))
    from_polars = sorted(pl.scan_delta(b).filter(cpl.predicate(b, {"root_id": roots})).collect().rows())
    assert from_duckdb == from_polars
    assert len(from_duckdb) == 6


def test_scan_delta_partitionwise_join(make_table):
    con = duckdb.connect()
    left = make_table("left", 16, synapses(100, seed=2))
    right = make_table("right", 32, synapses(100, seed=2))
    full = con.sql(
        f"SELECT count(*) FROM ({delta_scan(con, left).sql_query()}) l "
        f"JOIN ({delta_scan(con, right).sql_query()}) r USING (root_id)"
    ).fetchall()[0][0]
    total = 0
    for _, (l, r) in cdd.scan_delta_partitionwise_join(con, [left, right], on="root_id"):
        total += l.join(r, "root_id").count("*").fetchall()[0][0]
    assert total == full == 100 * 3 * 3


def test_native_write_and_scan(make_table):
    from deltalake import write_deltalake

    con = duckdb.connect()
    cdd.register(con)
    data = synapses()
    table = make_table("t", 16)
    con.register("synapses", data)
    query = f"SELECT *, {cdd.partition_columns(table)} FROM synapses"
    write_deltalake(table, con.sql(query), mode="append")
    # The same partition columns as the shortcut write.
    expected_table = make_table("expected", 16, data)
    assert sorted_rows(table.to_pyarrow_table()) == sorted_rows(expected_table.to_pyarrow_table())

    roots = data.column("root_id").to_pylist()[::150]
    uri = table.table_uri
    relation = con.sql(
        f"SELECT root_id, value FROM delta_scan('{uri}') WHERE {cdd.predicate(table, {'root_id': roots})}"
    )
    expected = data.filter(pa.compute.is_in(data.column("root_id"), pa.array(roots)))
    assert sorted_rows(relation) == sorted_rows(expected)
    n_buckets = len(cavebear.partition_values(table, {"root_id": roots})["root_id_partition"])
    assert files_read(con, relation) == n_buckets < 16


def test_expressions_take_a_spec():
    from conftest import spec

    assert cdd.partition_columns(spec(16)) == 'cave_iceberg_bucket("root_id", 16) AS "root_id_partition"'
    assert cdd.predicate(spec(16), {"root_id": [34]}) == '"root_id_partition" IN (3) AND "root_id" IN (34)'
    assert cdd.predicate(spec(16), {}) == "TRUE"


def test_partitionwise_join_predicates(make_table):
    fine, coarse = make_table("fine", 64), make_table("coarse", 16)
    steps = list(cdd.partitionwise_join_predicates([fine, coarse], on="root_id"))
    assert [value for value, _ in steps] == list(range(16))
    assert steps[3][1] == ['"root_id_partition" IN (3, 19, 35, 51)', '"root_id_partition" IN (3)']
