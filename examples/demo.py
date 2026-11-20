# %% [markdown]
# # Delta tables with a partition spec
#
# This demo shows how cavebear partitions ID columns by a bucket hash following the 
# Iceberg bucket strategy, which we reimplement for Delta Lake tables here.
#
# The demo has these steps:
#
# 1. Create two tables, partitioned with a bucket hash of the root ID.
# 2. Write the tables.
# 3. Read the synapses of a few roots, with file pruning.
#
# Most steps use the native calls of each engine (`write_deltalake`, `pl.scan_delta`,
# DuckDB's `delta_scan`), with cavebear expressions for the partition columns and the
# filters. Where cavebear also has a shortcut (`write_delta`), the demo does the step
# both ways. The shortcuts can change.

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
# A synthetic synapse table, and a table of features with one row per synapse. The
# feature table has the root ID of each synapse too, so that it can have the same
# partition key. The root IDs are random integers in the range of real CAVE root IDs.

# %%
rng = np.random.default_rng(0)
roots = np.unique(rng.integers(864691135_000_000_000, 864691137_000_000_000, 20_000))
n_synapses = 8_000_000

synapses = pl.DataFrame(
    {
        "id": np.arange(n_synapses),
        "pre_pt_root_id": rng.choice(roots, n_synapses),
        "post_pt_root_id": rng.choice(roots, n_synapses),
        "size": rng.integers(10, 5_000, n_synapses, dtype=np.int32),
    }
)
FEATURES = ["spine_volume", "distance_to_soma"]
features = synapses.select("id", "post_pt_root_id").with_columns(
    spine_volume=rng.lognormal(-2, 1, n_synapses),
    distance_to_soma=rng.uniform(0, 500, n_synapses),
).sample(fraction=1, shuffle=True, seed=0)
synapses.head()

# %%
features.head()

# %% [markdown]
# Each table gets a partition spec: a bucket of `post_pt_root_id`. The bucket hashes
# the root ID, and the hash modulo the bucket count gives the partition. The
# synapses have 64 partitions, and the features have 16. Tables that are written at
# different times, or that have different row sizes, often have different counts.
# The spec is stored as JSON in the Delta table `configuration`.

# %%
BUCKET = "post_pt_root_id_bucket"
synapse_spec = PartitionSpec((PartitionField(BUCKET, "post_pt_root_id", IcebergBucket(64)),))
feature_spec = PartitionSpec((PartitionField(BUCKET, "post_pt_root_id", IcebergBucket(16)),))
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

# %% [markdown]
# ## Polars
#
# ### 1. Create
#
# `create_delta` creates an empty Delta table, adds the partition columns to the
# schema, and puts the partition spec in the table configuration.

# %%
synapses_dt = cavebear.create_delta(root / "polars" / "synapses", synapses.schema, synapse_spec)
features_dt = cavebear.create_delta(root / "polars" / "features", features.schema, feature_spec)
synapses_dt.metadata().configuration

# %% [markdown]
# ### 2. Write
#
# `cpl.partition_columns` gives one Polars expression per partition field. Add them
# with `with_columns`, then write natively with `write_deltalake`.

# %%
features.with_columns(cpl.partition_columns(features_dt)).head()

# %%
write_deltalake(
    features_dt, features.with_columns(cpl.partition_columns(features_dt)), mode="append"
)
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
# The same partition filter also makes joins across tables efficient. If two tables
# are partitioned by the same key, a join can run one group of partitions at a time.
# Each step filters both tables to the partitions of that group, so it opens only
# their files and holds only their rows in memory. This works even if the two tables
# have different bucket counts. `partitionwise_join.py` shows this partition-wise
# join with Polars and with DuckDB.

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
features_dt = cavebear.create_delta(root / "duckdb" / "features", features.schema, feature_spec)

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
features_rel = con.from_arrow(features.to_arrow())
first_rel, second_rel = con.from_arrow(first.to_arrow()), con.from_arrow(second.to_arrow())
write_deltalake(
    features_dt, features_rel.project(f"*, {cdd.partition_columns(features_dt)}"), mode="append"
)
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
# %%
shutil.rmtree(root)
