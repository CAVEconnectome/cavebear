# %% [markdown]
# # Partition-wise joins of Delta tables with a partition spec
#
# Connectome tables are often too large to join in memory. A synapse table and a
# table of synapse features can each be hundreds of GB. This demo shows how cavebear
# lets you join such tables one piece at a time, with Polars and with DuckDB.
#
# The idea is to partition both tables by the same key, here the postsynaptic root
# ID. All rows of one root are then in one partition of each table. A join of one
# partition of each table is complete for the roots of that partition, so the join
# can run one partition at a time. This is a partition-wise join.
#
# The demo has these steps:
#
# 1. Create two tables, partitioned with a bucket hash of the root ID.
# 2. Write the tables.
# 3. Read the synapses of a few roots, with file pruning.
# 4. Join the two tables. First, two typical joins and how they fail. Then, the
#    partition-wise join.
#
# Most steps use the native calls of each engine (`write_deltalake`, `pl.scan_delta`,
# DuckDB's `delta_scan`), with cavebear expressions for the partition columns and the
# filters. Where cavebear also has a shortcut (`write_delta`, `scan_delta_partitionwise_join`),
# the demo does the step both ways. The shortcuts can change.

# %%
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from deltalake import DeltaTable, write_deltalake
from polars.testing import assert_frame_equal

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
# ### 4. Join
#
# The goal: for each root, the number of synapses and the mean of each feature. This
# needs a join of the synapses to the features on the synapse `id`, then an
# aggregation by root.


# %%
def by_root(joined: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate joined synapses and features by postsynaptic root."""
    return joined.group_by("post_pt_root_id").agg(
        pl.len().alias("n_synapses"), *(pl.col(f).mean() for f in FEATURES)
    )


def per_root(syn: pl.LazyFrame, feat: pl.LazyFrame) -> pl.LazyFrame:
    """Join synapses to their features, and aggregate by postsynaptic root."""
    return by_root(syn.join(feat.select("id", *FEATURES), on="id"))


# %% [markdown]
# #### The plain join
#
# The usual way is to scan both tables and join them. The result is correct.

# %%
plain = per_root(pl.scan_delta(synapses_dt), pl.scan_delta(features_dt)).collect()
assert plain["n_synapses"].sum() == n_synapses
plain.head()

# %% [markdown]
# But this join reads all files of both tables in one query. The join keeps one
# table in a hash table, so memory grows with the size of the whole table. The
# partition filter cannot help, because the query wants every root. Here the tables
# are small, but for larger data this can be inefficient.

# %%
for dt in (synapses_dt, features_dt):
    size = pl.scan_delta(dt).collect().estimated_size("mb")
    print(f"{Path(dt.table_uri).name}: {len(dt.file_uris())} files, {size:.0f} MB in memory")

# %% [markdown]
# #### Join partition by partition, by matching bucket numbers
#
# Both tables are partitioned by root, so a natural fix is a streaming loop: join 
# partition 0 of the synapses with partition 0 of the features, then partition 1 with 
# partition 1, and so on. Each step then holds only one partition of each table.
#
# With different bucket counts, this loop is wrong, and it gives no error. The
# features have only 16 partitions, so the loop has 16 steps. Synapse partitions 16
# to 63 never match a feature partition, and their roots are missing from the result.

# %%
naive = pl.concat(
    per_root(
        pl.scan_delta(synapses_dt).filter(pl.col(BUCKET) == v),
        pl.scan_delta(features_dt).filter(pl.col(BUCKET) == v),
    ).collect()
    for v in range(16)
)
print(f"{naive['n_synapses'].sum()} of {n_synapses} synapses, {naive.height} of {plain.height} roots")

# %% [markdown]
# The reason: one root has a different partition number in each table. Both
# partitions come from the same hash, so for a power-of-two count, the partition in
# the table with 16 buckets is the partition in the table with 64 buckets, modulo 16.

# %%
example_root = next(r for r in read_roots if IcebergBucket(64).bucket(r) >= 16)
in_synapses = IcebergBucket(64).bucket(example_root)
in_features = IcebergBucket(16).bucket(example_root)
print(f"root {example_root}: synapse partition {in_synapses}, feature partition {in_features}")
assert in_features == in_synapses % 16

# %% [markdown]
# #### The partition-wise join
#
# The fix is to match partitions by that rule. Feature partition 3 holds the same
# roots as synapse partitions 3, 19, 35 and 51, which are "nested" in it. So each
# step joins one feature partition with its 4 nested synapse partitions.
#
# `cpl.partitionwise_join_predicates` gives, for each step, one filter per table. It
# reads the partition spec of each table, so it knows both bucket counts. Step 3:

# %%
tables = [synapses_dt, features_dt]
steps = list(cpl.partitionwise_join_predicates(tables, on="post_pt_root_id"))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")

# %% [markdown]
# Each root is in exactly one step, so the per-root result of a step is final. The
# results of the steps only need a concat. They match the plain join.

# %%
# Put each filter on the native scan of its table.
partitionwise = pl.concat(
    per_root(*(pl.scan_delta(dt).filter(pred) for dt, pred in zip(tables, preds))).collect()
    for _, preds in steps
)
assert_frame_equal(partitionwise.sort("post_pt_root_id"), plain.sort("post_pt_root_id"))

# Or, the same scans with the shortcut.
shortcut = pl.concat(
    per_root(*scans).collect()
    for _, scans in cpl.scan_delta_partitionwise_join(tables, on="post_pt_root_id")
)
assert_frame_equal(shortcut.sort("post_pt_root_id"), plain.sort("post_pt_root_id"))
partitionwise.head()

# %% [markdown]
# #### Join first, then filter
#
# Polars can also build the join once, and filter the joined frame in each step.
# Predicate pushdown moves each part of a filter below the join, to the scan that
# has its columns. So the scans still read only the files of the step.
#
# Both tables have a column `post_pt_root_id_bucket`. After a join on `id`, the
# column of the features is `post_pt_root_id_bucket_right`. The filter of a step
# must test both columns, each with the partitions of its own table.
# `cavebear.partitionwise_join_values` gives these partitions as lists.

# %%
joined = pl.scan_delta(synapses_dt).join(pl.scan_delta(features_dt), on="id")
values = list(cavebear.partitionwise_join_values(tables, on="post_pt_root_id"))


def step_filter(synapse_filter: dict, feature_filter: dict) -> pl.Expr:
    """The filter of one step on the joined frame."""
    return pl.col(BUCKET).is_in(synapse_filter[BUCKET]) & pl.col(f"{BUCKET}_right").is_in(
        feature_filter[BUCKET]
    )


def scans(lf: pl.LazyFrame) -> None:
    """Print, for each scan in the optimized plan, its table, file count and filter."""
    for line in lf.explain().splitlines():
        found = re.search(r"SCAN \[\S*/(\w+)/[^,\]]*(?:, \.\.\. (\d+) other sources)?", line)
        if found:
            print(f"{found[1]}: {1 + int(found[2] or 0)} files")
        elif "SELECTION" in line:
            print("   " + line.strip())


value, (synapse_filter, feature_filter) = values[3]
print(f"step {value}: {step_filter(synapse_filter, feature_filter)}\n")
scans(joined.filter(step_filter(synapse_filter, feature_filter)))

# %% [markdown]
# The filter reaches both scans. Polars follows the `_right` name back to the
# column of the features table. Each scan reads only the files of the step: 4
# synapse partitions and 1 feature partition. A filter on the `% 16` of each bucket
# column, or a rename of the column before the join, gives the same plan.
#
# The filter must test both tables. Polars does not move a filter from one table to
# the other. With a filter on the synapse bucket only, the features scan reads all
# its files. The result is still correct, but each step reads the full features
# table.

# %%
scans(joined.filter(pl.col(BUCKET).is_in(synapse_filter[BUCKET])))

# %% [markdown]
# The bucket columns must be plain columns of the scans. A filter on a column that
# `with_columns` makes, for example `k = post_pt_root_id_bucket % 16`, stops at the
# `with_columns`, and then each scan reads all its files.
#
# With a filter on both tables, the steps give the same result as the partition-wise
# join above.

# %%
join_first = pl.concat(
    by_root(joined.filter(step_filter(*filters))).collect() for _, filters in values
)
assert_frame_equal(join_first.sort("post_pt_root_id"), plain.sort("post_pt_root_id"))

# %% [markdown]
# #### Why one `collect` per step
#
# The loop above calls `collect` once for each step. In Polars, a different way can
# look better: build all steps as lazy frames, put them in one query with
# `pl.concat`, and call `collect` once. That query has the correct plan. Each branch
# of its union reads only the files of its step. But it removes the benefit of the
# partition-wise join.
#
# Polars runs one query as one unit. With the streaming engine, which is the default
# in Polars 2, the peak memory of this query is almost that of the plain join. So
# the query does not free the data of one step before the next step starts.
#
# The test below measures this. It runs each query in a new Python process, and
# reads the time of the query and the peak memory of that process. Each query needs
# its own process, because the peak memory of a process never goes down.

# %%
SETUP = f"""
import polars as pl
from deltalake import DeltaTable
import cavebear
from cavebear import polars as cpl

FEATURES = {FEATURES!r}
BUCKET = {BUCKET!r}
tables = [DeltaTable(uri) for uri in {[dt.table_uri for dt in tables]!r}]
{inspect.getsource(by_root)}
{inspect.getsource(per_root)}
{inspect.getsource(step_filter)}
lazy_steps = [
    per_root(*scans)
    for _, scans in cpl.scan_delta_partitionwise_join(tables, on="post_pt_root_id")
]
joined = pl.scan_delta(tables[0]).join(pl.scan_delta(tables[1]), on="id")
values = list(cavebear.partitionwise_join_values(tables, on="post_pt_root_id"))
"""


def measure(code: str) -> tuple[float, float]:
    """Run ``SETUP``, then ``code``, in a new Python process.

    Give the time of ``code`` in seconds, and the peak memory of the process in MB.
    """
    script = f"""{SETUP}
import resource, time
start = time.perf_counter()
{code}
print(time.perf_counter() - start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    seconds, peak = run.stdout.split()[-2:]
    # macOS gives bytes, and Linux gives KiB.
    return float(seconds), int(peak) / 1e6 if sys.platform == "darwin" else int(peak) * 1024 / 1e6


queries = {
    "no query (imports and setup only)": "pass",
    "plain join": "per_root(*(pl.scan_delta(dt) for dt in tables)).collect()",
    "pl.concat of lazy steps, one collect": "pl.concat(lazy_steps).collect()",
    "the same, with parallel=False": "pl.concat(lazy_steps, parallel=False).collect()",
    "pl.collect_all of lazy steps": "pl.collect_all(lazy_steps)",
    "one collect per step, then pl.concat": "pl.concat([step.collect() for step in lazy_steps])",
    "join once, filter and collect per step": (
        "pl.concat([by_root(joined.filter(step_filter(*f))).collect() for _, f in values])"
    ),
}
print(f"{'query':<40} {'time':>7} {'peak memory':>12}")
for name, code in queries.items():
    seconds, peak = measure(code)
    print(f"{name:<40} {seconds:>6.2f}s {peak:>9.0f} MB")

# %% [markdown]
# Only the two loops with one `collect` per step keep the memory of one step. It
# does not matter if the filters are on the scans or on a join that you write once.
# The other ways to run the steps use almost as much memory as the plain join:
#
# - `pl.concat` of the lazy steps.
# - `pl.concat` with `parallel=False`. With the in-memory engine, this argument
#   lowers the peak memory, but with the streaming engine it does not.
# - `pl.collect_all` of the lazy steps.
#
# The loop costs little time. Each step is still a parallel query, so the steps
# use all cores, one step after the other.
#
# So with Polars, the loop over steps must stay in Python, with one `collect` in
# each step. Concatenate the collected results, not the lazy frames.

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
# ### 4. Partition-wise join
#
# The same steps as for Polars. `cdd.partitionwise_join_predicates` gives the same
# filters as SQL. Step 3:

# %%
tables = [synapses_dt, features_dt]
steps = list(cdd.partitionwise_join_predicates(tables, on="post_pt_root_id"))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")


# %%
def duckdb_per_root(syn: duckdb.DuckDBPyRelation, feat: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    """Join synapses to their features, and aggregate by postsynaptic root."""
    means = ", ".join(f"avg({f}) AS {f}" for f in FEATURES)
    return (
        syn.select("id, post_pt_root_id")
        .join(feat.select(f"id, {', '.join(FEATURES)}"), "id")
        .aggregate(f"post_pt_root_id, count(*)::UINT32 AS n_synapses, {means}", "post_pt_root_id")
        .pl()
    )


# Put each filter on the native scan of its table.
duckdb_partitionwise = pl.concat(
    duckdb_per_root(
        *(con.sql(f"FROM delta_scan('{dt.table_uri}')").filter(p) for dt, p in zip(tables, preds))
    )
    for _, preds in steps
)
# Or, the same scans with the shortcut.
duckdb_shortcut = pl.concat(
    duckdb_per_root(*scans)
    for _, scans in cdd.scan_delta_partitionwise_join(con, tables, on="post_pt_root_id")
)
# The same result as with Polars.
for df in (duckdb_partitionwise, duckdb_shortcut):
    assert_frame_equal(df.sort("post_pt_root_id"), plain.sort("post_pt_root_id"))
duckdb_partitionwise.head()

# %%
shutil.rmtree(root)
