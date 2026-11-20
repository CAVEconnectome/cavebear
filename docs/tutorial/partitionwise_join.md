# Partition-wise joins of Delta tables with a partition spec

[Download the notebook](partitionwise_join.ipynb) · [View the source script](https://github.com/CAVEconnectome/cavebear/blob/main/examples/partitionwise_join.py)

!!! danger "Work in progress"

    This API is a work in progress. We are still exploring how to balance an
    efficient query plan with an API that mirrors the lazy expressions of Polars.

Connectome tables are often too large to join in memory. A synapse table and a
table of synapse features can each be hundreds of GB. This example shows how
cavebear lets you join such tables one piece at a time, with Polars and with DuckDB.

The idea is to partition both tables by the same key, here the postsynaptic root
ID. All rows of one root are then in one partition of each table. A join of one
partition of each table is complete for the roots of that partition, so the join
can run one partition at a time. This is a partition-wise join.

This example starts from two written tables. For how to create, write and read
them, see `demo.py`. The steps here: two typical joins and how they fail, then the
partition-wise join.

The joins use the native scans of each engine (`pl.scan_delta`, DuckDB's
`delta_scan`), with cavebear filters. Where cavebear also has a shortcut
(`scan_delta_partitionwise_join`), the example does the step both ways. The
shortcuts can change.

```python
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
from polars.testing import assert_frame_equal

import cavebear
from cavebear import IcebergBucket, PartitionSpec
from cavebear import duckdb as cdd
from cavebear import polars as cpl

root = Path(tempfile.mkdtemp(prefix="cavebear-partitionwise-join-"))
```

## The data

A synthetic synapse table, and a table of features with one row per synapse. The
feature table has the root ID of each synapse too, so that it can have the same
partition key. The root IDs are random integers in the range of real CAVE root IDs.

```python
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
```

<div class="nb-output" markdown>

<div class="nb-html"><div><small>shape: (5, 4)</small><table class="dataframe"><thead><tr><th>id</th><th>pre_pt_root_id</th><th>post_pt_root_id</th><th>size</th></tr><tr><td>i64</td><td>i64</td><td>i64</td><td>i32</td></tr></thead><tbody><tr><td>0</td><td>864691136929386398</td><td>864691136540358433</td><td>4122</td></tr><tr><td>1</td><td>864691136018444775</td><td>864691135103953073</td><td>3114</td></tr><tr><td>2</td><td>864691136640382803</td><td>864691136945356460</td><td>3019</td></tr><tr><td>3</td><td>864691136387969085</td><td>864691135381934086</td><td>860</td></tr><tr><td>4</td><td>864691135181983164</td><td>864691135896229288</td><td>662</td></tr></tbody></table></div></div>

</div>

```python
features.head()
```

<div class="nb-output" markdown>

<div class="nb-html"><div><small>shape: (5, 4)</small><table class="dataframe"><thead><tr><th>id</th><th>post_pt_root_id</th><th>spine_volume</th><th>distance_to_soma</th></tr><tr><td>i64</td><td>i64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>3129899</td><td>864691136194057372</td><td>0.10106</td><td>261.237628</td></tr><tr><td>6663405</td><td>864691136726129803</td><td>1.49631</td><td>478.947168</td></tr><tr><td>2359581</td><td>864691136633759585</td><td>0.174421</td><td>122.33408</td></tr><tr><td>6302609</td><td>864691136349255449</td><td>0.175938</td><td>42.77822</td></tr><tr><td>7211515</td><td>864691135409021007</td><td>0.132908</td><td>323.418354</td></tr></tbody></table></div></div>

</div>

Each table gets a partition spec: a bucket of `post_pt_root_id`. The bucket hashes
the root ID, and the hash modulo the bucket count gives the partition. The
synapses have 64 partitions, and the features have 16. Tables that are written at
different times, or that have different row sizes, often have different counts.
The spec is stored as JSON in the Delta table `configuration`.

```python
synapse_spec = PartitionSpec({"post_pt_root_id": IcebergBucket(64)})
feature_spec = PartitionSpec({"post_pt_root_id": IcebergBucket(16)})
BUCKET = synapse_spec.fields[0].name  # the default name, "post_pt_root_id_partition"
synapse_spec.to_dict()
```

<div class="nb-output" markdown>

```text
{'@type': 'cave_partition_spec_v1',
 'fields': [{'name': 'post_pt_root_id_partition',
   'source': 'post_pt_root_id',
   'transform': {'@type': 'iceberg_bucket', 'n_buckets': 64}}]}
```

</div>

Create and write both tables. `write_delta` adds the partition columns.

```python
synapses_dt = cavebear.create_delta(root / "synapses", synapses.schema, synapse_spec)
features_dt = cavebear.create_delta(root / "features", features.schema, feature_spec)
cavebear.write_delta(synapses_dt, synapses)
cavebear.write_delta(features_dt, features)
read_roots = synapses["post_pt_root_id"].unique(maintain_order=True).to_list()[:3]
```

## Polars

### The goal

The goal: for each root, the number of synapses and the mean of each feature. This
needs a join of the synapses to the features on the synapse `id`, then an
aggregation by root.

```python
def by_root(joined: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate joined synapses and features by postsynaptic root."""
    return joined.group_by("post_pt_root_id").agg(
        pl.len().alias("n_synapses"), *(pl.col(f).mean() for f in FEATURES)
    )


def per_root(syn: pl.LazyFrame, feat: pl.LazyFrame) -> pl.LazyFrame:
    """Join synapses to their features, and aggregate by postsynaptic root."""
    return by_root(syn.join(feat.select("id", *FEATURES), on="id"))
```

#### The plain join

The usual way is to scan both tables and join them. The result is correct.

```python
plain = per_root(pl.scan_delta(synapses_dt), pl.scan_delta(features_dt)).collect()
assert plain["n_synapses"].sum() == n_synapses
plain.head()
```

<div class="nb-output" markdown>

<div class="nb-html"><div><small>shape: (5, 4)</small><table class="dataframe"><thead><tr><th>post_pt_root_id</th><th>n_synapses</th><th>spine_volume</th><th>distance_to_soma</th></tr><tr><td>i64</td><td>u32</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>864691136124772776</td><td>403</td><td>0.218049</td><td>252.428983</td></tr><tr><td>864691135996558641</td><td>417</td><td>0.224784</td><td>251.597761</td></tr><tr><td>864691135047855631</td><td>368</td><td>0.234856</td><td>251.117611</td></tr><tr><td>864691136672595914</td><td>393</td><td>0.196389</td><td>236.446979</td></tr><tr><td>864691135923432826</td><td>391</td><td>0.225052</td><td>253.046563</td></tr></tbody></table></div></div>

</div>

But this join reads all files of both tables in one query. The join keeps one
table in a hash table, so memory grows with the size of the whole table. The
partition filter cannot help, because the query wants every root. Here the tables
are small, but for larger data this can be inefficient.

```python
for dt in (synapses_dt, features_dt):
    size = pl.scan_delta(dt).collect().estimated_size("mb")
    print(f"{Path(dt.table_uri).name}: {len(dt.file_uris())} files, {size:.0f} MB in memory")
```

<div class="nb-output" markdown>

```text
synapses: 64 files, 244 MB in memory
features: 16 files, 275 MB in memory
```

</div>

#### Join partition by partition, by matching bucket numbers

Both tables are partitioned by root, so a natural fix is a streaming loop: join 
partition 0 of the synapses with partition 0 of the features, then partition 1 with 
partition 1, and so on. Each step then holds only one partition of each table.

With different bucket counts, this loop is wrong, and it gives no error. The
features have only 16 partitions, so the loop has 16 steps. Synapse partitions 16
to 63 never match a feature partition, and their roots are missing from the result.

```python
naive = pl.concat(
    per_root(
        pl.scan_delta(synapses_dt).filter(pl.col(BUCKET) == v),
        pl.scan_delta(features_dt).filter(pl.col(BUCKET) == v),
    ).collect()
    for v in range(16)
)
print(f"{naive['n_synapses'].sum()} of {n_synapses} synapses, {naive.height} of {plain.height} roots")
```

<div class="nb-output" markdown>

```text
2015369 of 8000000 synapses, 5038 of 20000 roots
```

</div>

The reason: one root has a different partition number in each table. Both
partitions come from the same hash, so for a power-of-two count, the partition in
the table with 16 buckets is the partition in the table with 64 buckets, modulo 16.

```python
example_root = next(r for r in read_roots if IcebergBucket(64).bucket(r) >= 16)
in_synapses = IcebergBucket(64).bucket(example_root)
in_features = IcebergBucket(16).bucket(example_root)
print(f"root {example_root}: synapse partition {in_synapses}, feature partition {in_features}")
assert in_features == in_synapses % 16
```

<div class="nb-output" markdown>

```text
root 864691136945356460: synapse partition 47, feature partition 15
```

</div>

#### The partition-wise join

The fix is to match partitions by that rule. Feature partition 3 holds the same
roots as synapse partitions 3, 19, 35 and 51, which are "nested" in it. So each
step joins one feature partition with its 4 nested synapse partitions.

`cpl.partitionwise_join_predicates` gives, for each step, one filter per table. It
reads the partition spec of each table, so it knows both bucket counts. Step 3:

```python
tables = [synapses_dt, features_dt]
steps = list(cpl.partitionwise_join_predicates(tables, on="post_pt_root_id"))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")
```

<div class="nb-output" markdown>

```text
step 3, synapses: col("post_pt_root_id_partition").is_in([[3, 19, … 51]])
step 3, features: col("post_pt_root_id_partition").is_in([[3]])
```

</div>

Each root is in exactly one step, so the per-root result of a step is final. The
results of the steps only need a concat. They match the plain join.

```python
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
```

<div class="nb-output" markdown>

<div class="nb-html"><div><small>shape: (5, 4)</small><table class="dataframe"><thead><tr><th>post_pt_root_id</th><th>n_synapses</th><th>spine_volume</th><th>distance_to_soma</th></tr><tr><td>i64</td><td>u32</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>864691135329723005</td><td>379</td><td>0.18885</td><td>263.305268</td></tr><tr><td>864691136114317759</td><td>440</td><td>0.229895</td><td>260.579551</td></tr><tr><td>864691136767509611</td><td>397</td><td>0.246031</td><td>252.843313</td></tr><tr><td>864691136152182991</td><td>419</td><td>0.211741</td><td>250.828895</td></tr><tr><td>864691136853742121</td><td>391</td><td>0.229737</td><td>247.919078</td></tr></tbody></table></div></div>

</div>

#### Join first, then filter

Polars can also build the join once, and filter the joined frame in each step.
Predicate pushdown moves each part of a filter below the join, to the scan that
has its columns. So the scans still read only the files of the step.

Both tables have a column `post_pt_root_id_partition`. After a join on `id`, the
column of the features is `post_pt_root_id_partition_right`. The filter of a step
must test both columns, each with the partitions of its own table.
`cavebear.partitionwise_join_values` gives these partitions as lists.

```python
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
```

<div class="nb-output" markdown>

```text
step 3: [(col("post_pt_root_id_partition").is_in([[3, 19, … 51]])) & (col("post_pt_root_id_partition_right").is_in([[3]]))]

synapses: 4 files
   SELECTION: col("post_pt_root_id_partition").is_in([[3, 19, … 51]])
features: 1 files
   SELECTION: col("post_pt_root_id_partition").alias("post_pt_root_id_partition_right").is_in([[3]])
```

</div>

The filter reaches both scans. Polars follows the `_right` name back to the
column of the features table. Each scan reads only the files of the step: 4
synapse partitions and 1 feature partition. A filter on the `% 16` of each bucket
column, or a rename of the column before the join, gives the same plan.

The filter must test both tables. Polars does not move a filter from one table to
the other. With a filter on the synapse bucket only, the features scan reads all
its files. The result is still correct, but each step reads the full features
table.

```python
scans(joined.filter(pl.col(BUCKET).is_in(synapse_filter[BUCKET])))
```

<div class="nb-output" markdown>

```text
synapses: 4 files
   SELECTION: col("post_pt_root_id_partition").is_in([[3, 19, … 51]])
features: 16 files
```

</div>

The bucket columns must be plain columns of the scans. A filter on a column that
`with_columns` makes, for example `k = post_pt_root_id_partition % 16`, stops at the
`with_columns`, and then each scan reads all its files.

With a filter on both tables, the steps give the same result as the partition-wise
join above.

```python
join_first = pl.concat(
    by_root(joined.filter(step_filter(*filters))).collect() for _, filters in values
)
assert_frame_equal(join_first.sort("post_pt_root_id"), plain.sort("post_pt_root_id"))
```

#### Why one `collect` per step

The loop above calls `collect` once for each step. In Polars, a different way can
look better: build all steps as lazy frames, put them in one query with
`pl.concat`, and call `collect` once. That query has the correct plan. Each branch
of its union reads only the files of its step. But it removes the benefit of the
partition-wise join.

Polars runs one query as one unit. With the streaming engine, which is the default
in Polars 2, the peak memory of this query is almost that of the plain join. So
the query does not free the data of one step before the next step starts.

The test below measures this. It runs each query in a new Python process, and
reads the time of the query and the peak memory of that process. Each query needs
its own process, because the peak memory of a process never goes down.

```python
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
```

<div class="nb-output" markdown>

```text
query                                       time  peak memory
no query (imports and setup only)          0.00s        99 MB
plain join                                 0.30s      1024 MB
pl.concat of lazy steps, one collect       0.15s       851 MB
the same, with parallel=False              0.16s       861 MB
pl.collect_all of lazy steps               0.23s       844 MB
one collect per step, then pl.concat       0.24s       424 MB
join once, filter and collect per step     0.25s       437 MB
```

</div>

Only the two loops with one `collect` per step keep the memory of one step. It
does not matter if the filters are on the scans or on a join that you write once.
The other ways to run the steps use almost as much memory as the plain join:

- `pl.concat` of the lazy steps.
- `pl.concat` with `parallel=False`. With the in-memory engine, this argument
  lowers the peak memory, but with the streaming engine it does not.
- `pl.collect_all` of the lazy steps.

The loop costs little time. Each step is still a parallel query, so the steps
use all cores, one step after the other.

So with Polars, the loop over steps must stay in Python, with one `collect` in
each step. Concatenate the collected results, not the lazy frames.

## DuckDB

The same partition-wise join with DuckDB, on the same tables. The cavebear
functions have the same names as for Polars, but they give SQL text, which works
in a query and in the relation API. `cdd.partitionwise_join_predicates` gives the
same filters as `cpl.partitionwise_join_predicates`, as SQL. Step 3:

```python
con = duckdb.connect()
# filters as SQL. Step 3:
```

```python
tables = [synapses_dt, features_dt]
steps = list(cdd.partitionwise_join_predicates(tables, on="post_pt_root_id"))
value, preds = steps[3]
for dt, pred in zip(tables, preds):
    print(f"step {value}, {Path(dt.table_uri).name}: {pred}")
```

<div class="nb-output" markdown>

```text
step 3, synapses: "post_pt_root_id_partition" IN (3, 19, 35, 51)
step 3, features: "post_pt_root_id_partition" IN (3)
```

</div>

```python
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
```

<div class="nb-output" markdown>

<div class="nb-html"><div><small>shape: (5, 4)</small><table class="dataframe"><thead><tr><th>post_pt_root_id</th><th>n_synapses</th><th>spine_volume</th><th>distance_to_soma</th></tr><tr><td>i64</td><td>u32</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>864691135248566552</td><td>396</td><td>0.233946</td><td>262.81106</td></tr><tr><td>864691135917760129</td><td>403</td><td>0.226536</td><td>239.580864</td></tr><tr><td>864691135465544769</td><td>400</td><td>0.227324</td><td>240.441258</td></tr><tr><td>864691135010407556</td><td>414</td><td>0.209746</td><td>261.93996</td></tr><tr><td>864691135775742944</td><td>409</td><td>0.199426</td><td>243.622401</td></tr></tbody></table></div></div>

</div>

```python
shutil.rmtree(root)
```
