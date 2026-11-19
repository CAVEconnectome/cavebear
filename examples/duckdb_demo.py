"""Write and read a Delta table with a partition spec from DuckDB.

Part 1 uses the low-level API: the native ``write_deltalake`` call and DuckDB's
``delta_scan``, with the partition columns and the filter from cavebear SQL. Part 2
uses the high-level API of cavebear (``write``, ``scan`` and ``scan_partitionwise``),
which can change.

Run it with ``uv run python examples/duckdb_demo.py``.
"""

import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
from deltalake import write_deltalake
from synthetic import (
    CELL_SCHEMA,
    CELL_SPEC,
    SYNAPSE_SCHEMA,
    SYNAPSE_SPEC,
    files_opened,
    sorted_rows,
    synthetic_tables,
)

import cavebear
from cavebear import duckdb as cdd

COLUMNS = SYNAPSE_SCHEMA.names


def low_level(con: duckdb.DuckDBPyConnection, root: Path, roots: list[int]) -> pa.Table:
    print("Part 1: low-level API")

    # cavebear creates the table, so that the partition spec is in its configuration.
    dt = cavebear.create_delta(root / "synapses_low", SYNAPSE_SCHEMA, SYNAPSE_SPEC)

    # Write: register the bucket function, add the partition columns in SQL, then
    # write the relation natively.
    cdd.register(con)
    columns = cdd.partition_columns(dt)
    print(f"  partition columns: {columns}")
    write_deltalake(dt, con.sql(f"SELECT *, {columns} FROM synapses"), mode="append")
    print("  wrote the synapses with write_deltalake")

    # Read: a native delta_scan with a cavebear predicate. The predicate tests the
    # partition column, so that DuckDB opens only the files of those partitions, and
    # the source column, so that only the rows of those roots stay.
    condition = cdd.predicate(dt, {"post_pt_root_id": roots})
    print(f"  predicate: {condition}")
    result = con.sql(
        f"SELECT * FROM delta_scan('{dt.table_uri}', version => {dt.version()}) WHERE {condition}"
    ).to_arrow_table()
    print(f"  read {result.num_rows} rows with delta_scan")
    print(f"  {files_opened(dt, 'post_pt_root_id', roots)}")
    return result


def high_level(con: duckdb.DuckDBPyConnection, root: Path, roots: list[int]) -> pa.Table:
    print("Part 2: high-level API")

    # Write: cavebear.write takes any relation, adds the partition columns and appends.
    dt = cavebear.write(
        cavebear.create_delta(root / "synapses", SYNAPSE_SCHEMA, SYNAPSE_SPEC),
        con.sql("SELECT * FROM synapses"),
    )
    cells_dt = cavebear.write(
        cavebear.create_delta(root / "cells", CELL_SCHEMA, CELL_SPEC), con.sql("SELECT * FROM cells")
    )
    print("  wrote the synapses and the cells with cavebear.write")

    # Read: cdd.scan is delta_scan at the version of dt, with the predicate of part 1.
    result = cdd.scan(con, dt, {"post_pt_root_id": roots}).to_arrow_table()
    print(f"  read {result.num_rows} rows with cdd.scan")

    # A partition-wise join runs one partition of bucket[16] at a time. Each step reads
    # one partition of the cell table and the 4 synapse partitions nested in it.
    counts: dict[str, int] = {}
    steps = 0
    for _, (syn, cell) in cdd.scan_partitionwise(con, [dt, cells_dt], on=["post_pt_root_id", "pt_root_id"]):
        rows = con.sql(
            f"SELECT c.cell_type, count(*) FROM ({syn.sql_query()}) s "
            f"JOIN ({cell.sql_query()}) c ON s.post_pt_root_id = c.pt_root_id GROUP BY c.cell_type"
        ).fetchall()
        for cell_type, n in rows:
            counts[cell_type] = counts.get(cell_type, 0) + n
        steps += 1
    print(f"  partition-wise join in {steps} steps, synapses by postsynaptic cell type:")
    for cell_type in sorted(counts):
        print(f"    {cell_type}: {counts[cell_type]}")
    assert sum(counts.values()) == con.sql("SELECT count(*) FROM synapses").fetchall()[0][0]
    return result


def main(root: Path) -> None:
    synapses, cells = synthetic_tables()
    con = duckdb.connect()
    con.register("synapses", synapses)
    con.register("cells", cells)
    roots = synapses.column("post_pt_root_id").unique().to_pylist()[:3]
    keep = pa.compute.is_in(synapses.column("post_pt_root_id"), pa.array(roots))
    expected = synapses.filter(keep)
    print(f"Reading post_pt_root_id in {roots}: {expected.num_rows} rows expected\n")

    low = low_level(con, root, roots)
    print()
    high = high_level(con, root, roots)
    for result in (low, high):
        assert sorted_rows(result, COLUMNS) == sorted_rows(expected, COLUMNS)
    print("\nBoth parts return the expected rows.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cavebear-duckdb-demo-") as tmp:
        main(Path(tmp))
