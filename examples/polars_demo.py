"""Write and read Delta tables with a partition spec from Polars.

Both parts do the same steps on two tables, synapses with 64 partitions and cells
with 16 partitions:

1. Create the two tables.
2. Write the cells, then append the synapses in two batches.
3. Read the synapses of a few roots.
4. Join synapses to cells partition-wise, and count synapses by cell type.

Part 1 uses the low-level API: the native ``write_deltalake`` and ``pl.scan_delta``
calls, with cavebear expressions for the partition columns and the filters. Part 2
uses the high-level API of cavebear (``write`` and ``scan_partitionwise``), which can
change. The high-level API has no read function, so both parts read the same way.

Run it with ``uv run python examples/polars_demo.py``.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl
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
from cavebear import polars as cpl

ON = ["post_pt_root_id", "pt_root_id"]


def count_by_type(steps: Iterator[tuple[pl.LazyFrame, pl.LazyFrame]]) -> pl.DataFrame:
    """Join each step of a partition-wise join, and count synapses by cell type."""
    counts = [
        syn.join(cell, left_on=ON[0], right_on=ON[1]).group_by("cell_type").len().collect()
        for syn, cell in steps
    ]
    print(f"  partition-wise join in {len(counts)} steps")
    return pl.concat(counts).group_by("cell_type").agg(pl.col("len").sum()).sort("cell_type")


def low_level(
    root: Path, batches: list[pl.DataFrame], cells: pl.DataFrame, roots: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Part 1: low-level API")

    # 1. Create. cavebear puts the partition spec in the table configuration.
    synapses_dt = cavebear.create_delta(root / "low" / "synapses", SYNAPSE_SCHEMA, SYNAPSE_SPEC)
    cells_dt = cavebear.create_delta(root / "low" / "cells", CELL_SCHEMA, CELL_SPEC)

    # 2. Write. Add the partition columns with Polars expressions, then write natively.
    write_deltalake(cells_dt, cells.with_columns(cpl.partition_columns(cells_dt)), mode="append")
    for batch in batches:
        write_deltalake(synapses_dt, batch.with_columns(cpl.partition_columns(synapses_dt)), mode="append")
    print(f"  wrote the cells, then appended {len(batches)} synapse batches, with write_deltalake")

    # 3. Read. The predicate tests the partition column, so that Polars opens only the
    # files of those partitions, and the source column, so that only the rows of those
    # roots stay.
    where = {"post_pt_root_id": roots}
    result = pl.scan_delta(synapses_dt).filter(cpl.predicate(synapses_dt, where)).collect()
    print(f"  read {result.height} rows with pl.scan_delta")
    print(f"  {files_opened(synapses_dt, 'post_pt_root_id', roots)}")

    # 4. Join. iter_partitionwise gives the partition values of each table for each
    # step: one partition of the cells, and the 4 synapse partitions nested in it.
    tables = [synapses_dt, cells_dt]
    steps = (
        [
            pl.scan_delta(dt).filter(*(pl.col(name).is_in(values) for name, values in f.items()))
            for dt, f in zip(tables, filters)
        ]
        for _, filters in cavebear.iter_partitionwise(tables, on=ON)
    )
    return result, count_by_type(steps)


def high_level(
    root: Path, batches: list[pl.DataFrame], cells: pl.DataFrame, roots: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Part 2: high-level API")

    # 1. Create.
    synapses_dt = cavebear.create_delta(root / "high" / "synapses", SYNAPSE_SCHEMA, SYNAPSE_SPEC)
    cells_dt = cavebear.create_delta(root / "high" / "cells", CELL_SCHEMA, CELL_SPEC)

    # 2. Write. cavebear.write adds the partition columns and appends.
    cavebear.write(cells_dt, cells)
    for batch in batches:
        cavebear.write(synapses_dt, batch)
    print(f"  wrote the cells, then appended {len(batches)} synapse batches, with cavebear.write")

    # 3. Read. There is no high-level read: the native scan and the predicate of part 1
    # are already one line.
    where = {"post_pt_root_id": roots}
    result = pl.scan_delta(synapses_dt).filter(cpl.predicate(synapses_dt, where)).collect()
    print(f"  read {result.height} rows with pl.scan_delta")

    # 4. Join. cpl.scan_partitionwise gives the filtered scans of each step.
    steps = (scans for _, scans in cpl.scan_partitionwise([synapses_dt, cells_dt], on=ON))
    return result, count_by_type(steps)


def main(root: Path) -> None:
    synapses, cells = (pl.from_arrow(t) for t in synthetic_tables())
    half = synapses.height // 2
    batches = [synapses[:half], synapses[half:]]
    roots = synapses["post_pt_root_id"].unique(maintain_order=True).to_list()[:3]
    expected = synapses.filter(pl.col("post_pt_root_id").is_in(roots))
    print(f"Reading post_pt_root_id in {roots}: {expected.height} rows expected\n")

    low_rows, low_counts = low_level(root, batches, cells, roots)
    print()
    high_rows, high_counts = high_level(root, batches, cells, roots)

    columns = SYNAPSE_SCHEMA.names
    for rows in (low_rows, high_rows):
        assert sorted_rows(rows.to_arrow(), columns) == sorted_rows(expected.to_arrow(), columns)
    assert low_counts.equals(high_counts)
    assert low_counts["len"].sum() == synapses.height
    print("\nSynapses by postsynaptic cell type, the same in both parts:")
    for cell_type, n in low_counts.iter_rows():
        print(f"  {cell_type}: {n}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cavebear-polars-demo-") as tmp:
        main(Path(tmp))
