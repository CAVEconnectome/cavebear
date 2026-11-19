# %% [markdown]
# # Delta tables with a partition spec, from Polars and DuckDB
#
# This demo writes two Delta tables that are partitioned with the `iceberg_bucket`
# transform, and reads them with file pruning. It does the same steps first with
# Polars, then with DuckDB:
#
# 1. Create the two tables.
# 2. Write the cells, then append the synapses in two batches.
# 3. Read the synapses of a few roots.
# 4. Join synapses to cells partition-wise, and count synapses by cell type.
#
# Most steps use the native calls of each engine (`write_deltalake`, `pl.scan_delta`,
# DuckDB's `delta_scan`), with cavebear expressions for the partition columns and the
# filters. Where cavebear also has a shortcut (`write_delta`, `scan_delta_partitionwise_join`),
# the demo does the step both ways. The shortcuts can change.

# %%
import shutil
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from deltalake import DeltaTable, write_deltalake

import cavebear
from cavebear import IcebergBucket, PartitionField, PartitionSpec
from cavebear import duckdb as cdd
from cavebear import polars as cpl

root = Path(tempfile.mkdtemp(prefix="cavebear-demo-"))

# %% [markdown]
# ## The data
#
# A synthetic synapse table, and a cell table with one cell type for each root ID.
# The root IDs are random integers in the range of real CAVE root IDs.

# %%
rng = np.random.default_rng(0)
roots = np.unique(rng.integers(864691135_000_000_000, 864691137_000_000_000, 2_000))
n_synapses = 200_000

synapses = pl.DataFrame(
    {
        "id": np.arange(n_synapses),
        "pre_pt_root_id": rng.choice(roots, n_synapses),
        "post_pt_root_id": rng.choice(roots, n_synapses),
        "size": rng.integers(10, 5_000, n_synapses, dtype=np.int32),
    }
)
cells = pl.DataFrame(
    {"pt_root_id": roots, "cell_type": rng.choice(["excitatory", "inhibitory"], len(roots))}
)
synapses.head()

# %%
cells.head()

# %% [markdown]
# Each table gets a partition spec: a bucket of a root ID column. The synapses have
# 64 partitions on `post_pt_root_id`, and the cells have 16 partitions on
# `pt_root_id`. The spec is stored as JSON in the Delta table `configuration`.

# %%
synapse_spec = PartitionSpec(
    (PartitionField("post_pt_root_id_bucket", "post_pt_root_id", IcebergBucket(64)),)
)
cell_spec = PartitionSpec((PartitionField("pt_root_id_bucket", "pt_root_id", IcebergBucket(16)),))
synapse_spec.to_dict()

# %% [markdown]
# Two helpers for the checks below. `files_opened` counts the files that a read of
# some roots opens, with and without the partition filter. Without it, Delta can skip
# a file only by the min and max root ID of the file, and the hash puts IDs from the
# whole range in every file.


# %%
def files_opened(dt: DeltaTable, source: str, values: list[int]) -> str:
    total = len(dt.file_uris())
    plain = len(dt.file_uris(file_pruning_predicate=[(source, "in", values)]))
    partitions = cavebear.partition_values(dt, {source: values})
    pruned = len(dt.file_uris(file_pruning_predicate=[(n, "in", v) for n, v in partitions.items()]))
    return f"{pruned} of {total} files (without the partition filter: {plain} of {total})"


def sorted_rows(df: pl.DataFrame) -> list[tuple]:
    return sorted(df.select(synapses.columns).rows())


half = n_synapses // 2
first, second = synapses[:half], synapses[half:]
read_roots = synapses["post_pt_root_id"].unique(maintain_order=True).to_list()[:3]
expected = synapses.filter(pl.col("post_pt_root_id").is_in(read_roots))
ON = ["post_pt_root_id", "pt_root_id"]

# %% [markdown]
# ## Polars
#
# ### 1. Create
#
# `create_delta` creates an empty Delta table, adds the partition columns to the
# schema, and puts the partition spec in the table configuration.

# %%
synapses_dt = cavebear.create_delta(root / "polars" / "synapses", synapses.schema, synapse_spec)
cells_dt = cavebear.create_delta(root / "polars" / "cells", cells.schema, cell_spec)
synapses_dt.metadata().configuration

# %% [markdown]
# ### 2. Write
#
# `cpl.partition_columns` gives one Polars expression per partition field. Add them
# with `with_columns`, then write natively with `write_deltalake`.

# %%
cells.with_columns(cpl.partition_columns(cells_dt)).head()

# %%
write_deltalake(cells_dt, cells.with_columns(cpl.partition_columns(cells_dt)), mode="append")
write_deltalake(synapses_dt, first.with_columns(cpl.partition_columns(synapses_dt)), mode="append")
# Or, the same append with a shortcut, which adds the partition columns itself.
cavebear.write_delta(synapses_dt, second)

# %% [markdown]
# ### 3. Read
#
# `cpl.predicate` gives a filter on the partition column, so that Polars opens only
# the files of those partitions, and on the source column, so that only the rows of
# those roots stay. The roots have rows from both appends.

# %%
where = {"post_pt_root_id": read_roots}
cpl.predicate(synapses_dt, where)

# %%
result = pl.scan_delta(synapses_dt).filter(cpl.predicate(synapses_dt, where)).collect()
assert sorted_rows(result) == sorted_rows(expected)
print(f"{result.height} rows, {files_opened(synapses_dt, 'post_pt_root_id', read_roots)}")
result.head()

# %% [markdown]
# ### 4. Partition-wise join
#
# The join runs one cell partition at a time. Both tables use the same hash, and the
# bucket counts are powers of 2 (64 for synapses, 16 for cells), so for every root,
# cell partition = synapse partition % 16. Cell partition 3 therefore holds the same
# roots as synapse partitions 3, 19, 35 and 51, which are "nested" in it. Each step
# joins one cell partition with its 4 nested synapse partitions.
#
# `cpl.partitionwise_join_predicates` gives, for each step, one filter per table. Step 3:

# %%
tables = [synapses_dt, cells_dt]
steps = list(cpl.partitionwise_join_predicates(tables, on=ON))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")


# %%
def count_by_type(scans) -> pl.DataFrame:
    """Join each step of a partition-wise join, and count synapses by cell type."""
    counts = [
        syn.join(cell, left_on=ON[0], right_on=ON[1]).group_by("cell_type").len().collect()
        for syn, cell in scans
    ]
    return pl.concat(counts).group_by("cell_type").agg(pl.col("len").sum()).sort("cell_type")


# Put each filter on the native scan of its table.
native_scans = (
    [pl.scan_delta(dt).filter(pred) for dt, pred in zip(tables, preds)] for _, preds in steps
)
by_type = count_by_type(native_scans)
# Or, the same scans with the shortcut.
shortcut_scans = (scans for _, scans in cpl.scan_delta_partitionwise_join(tables, on=ON))
assert count_by_type(shortcut_scans).equals(by_type)
assert by_type["len"].sum() == n_synapses
by_type

# %% [markdown]
# ## DuckDB
#
# The same steps with DuckDB. The cavebear functions have the same names as for
# Polars, but they give SQL text, which works in a query and in the relation API.
#
# ### 1. Create
#
# Creating a table does not use an engine, so this step is the same as for Polars.

# %%
con = duckdb.connect()
synapses_dt = cavebear.create_delta(root / "duckdb" / "synapses", synapses.schema, synapse_spec)
cells_dt = cavebear.create_delta(root / "duckdb" / "cells", cells.schema, cell_spec)

# %% [markdown]
# ### 2. Write
#
# DuckDB has no bucket function, so `cdd.register` adds one to the connection, as a
# Python function. `cdd.partition_columns` gives a SQL select list that calls it.
# Add the list with `project`, then write natively with `write_deltalake`.

# %%
cdd.register(con)
cdd.partition_columns(synapses_dt)

# %%
cells_rel = con.from_arrow(cells.to_arrow())
first_rel, second_rel = con.from_arrow(first.to_arrow()), con.from_arrow(second.to_arrow())
write_deltalake(cells_dt, cells_rel.project(f"*, {cdd.partition_columns(cells_dt)}"), mode="append")
write_deltalake(
    synapses_dt, first_rel.project(f"*, {cdd.partition_columns(synapses_dt)}"), mode="append"
)
# Or, the same append with the shortcut, which adds the partition columns itself.
cavebear.write_delta(synapses_dt, second_rel)

# %% [markdown]
# ### 3. Read
#
# `cdd.predicate` gives the same filter as `cpl.predicate`, as SQL. The values are
# literal `IN` lists, so that DuckDB pushes them into `delta_scan` and skips files.

# %%
cdd.predicate(synapses_dt, where)

# %%
scan = con.sql(f"FROM delta_scan('{synapses_dt.table_uri}')")
result = scan.filter(cdd.predicate(synapses_dt, where)).pl()
assert sorted_rows(result) == sorted_rows(expected)
print(f"{result.height} rows, {files_opened(synapses_dt, 'post_pt_root_id', read_roots)}")
result.head()

# %% [markdown]
# ### 4. Partition-wise join
#
# The same steps as for Polars. `cdd.partitionwise_join_predicates` gives the same filters
# as SQL. Step 3:

# %%
tables = [synapses_dt, cells_dt]
steps = list(cdd.partitionwise_join_predicates(tables, on=ON))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")


# %%
def duckdb_count_by_type(scans) -> pl.DataFrame:
    """Join each step of a partition-wise join, and count synapses by cell type."""
    counts = [
        syn.set_alias("s")
        .join(cell.set_alias("c"), "s.post_pt_root_id = c.pt_root_id")
        .aggregate("cell_type, count(*) AS len", "cell_type")
        .pl()
        for syn, cell in scans
    ]
    return pl.concat(counts).group_by("cell_type").agg(pl.col("len").sum()).sort("cell_type")


# Put each filter on the native scan of its table.
native_scans = (
    [con.sql(f"FROM delta_scan('{dt.table_uri}')").filter(pred) for dt, pred in zip(tables, preds)]
    for _, preds in steps
)
duckdb_by_type = duckdb_count_by_type(native_scans)
# Or, the same scans with the shortcut.
shortcut_scans = (scans for _, scans in cdd.scan_delta_partitionwise_join(con, tables, on=ON))
assert duckdb_count_by_type(shortcut_scans).equals(duckdb_by_type)
# The same counts as with Polars.
assert duckdb_by_type.rows() == by_type.rows()
duckdb_by_type

# %%
shutil.rmtree(root)
