# Table layout

> Status: draft.

A _cavebear table_ is a Delta table whose rows are partitioned by deterministic
transforms of one or more source columns, for example a hash bucket of a segment ID.
Delta records which columns a table is partitioned by, but not how those partition
values were computed. The _layout_ of a cavebear table records that information, so
that any reader can compute the partition value for a given source value and use it
for file pruning or bucket-aligned joins.

The layout is table-level: every file in the table shares one layout. Changing the
layout requires rewriting the table.

## Storage

The layout is stored as a JSON file at `_info/layout.json`, relative to the Delta
table root:

```
<table root>/
  _delta_log/
  _info/
    layout.json
  <column_1>=<bucket_1>/
    <column_2>=<bucket_2>/
      part-*.parquet
```

- The `_info/` directory is reserved for cavebear metadata. The leading underscore
  follows the Delta and Hive convention for paths that are not table data.
- `_info/layout.json` is the single source of truth for the layout. Layout metadata
  must not also be stored in the Delta table `configuration`.
- A Delta table without `_info/layout.json` has no cavebear layout.

## Layout specification

`layout.json` contains a JSON object with the following members:

- `"@type"`: Must be `"cave_layout_v1"`.
- `"transforms"`: A non-empty array of _partition field_ objects, described below.
  Each entry describes one transformed Delta partition column, and no two entries may
  share the same `"name"`. The order of entries has no meaning; Delta's
  `partitionColumns` defines the order of partition columns. Partition columns not
  named by any entry are ordinary Delta partition columns, and the layout does not
  describe them.

Each partition field object has the following members:

- `"name"`: String. The name of the Delta partition column that stores the bucket
  value. It must appear in the Delta table's `partitionColumns`.
- `"source"`: String. The name of the column the bucket value is computed from. It
  must be a column of the table with a type accepted by the transform. As in
  Iceberg, more than one partition field may share the same source.
  A later transform with more than one source column, such as a spatial transform
  of `x`, `y` and `z`, will use a `"sources"` array instead of `"source"`. Iceberg
  v3 does the same with `source-id` and `source-ids`.
- `"transform"`: A _transform_ object. Its `"@type"` member identifies the transform,
  and its remaining members are defined by that transform. Must be one of:
  - `"iceberg_bucket"`: see [iceberg_bucket.md](./iceberg_bucket.md).
  - `"neuroglancer_uint64_sharded_v1"`: see
    [neuroglancer_sharded.md](./neuroglancer_sharded.md).

Readers must reject a layout whose top-level `"@type"` they do not recognize. As in
Iceberg, a reader that does not recognize a transform's `"@type"` must ignore that
partition field, and must not use it for file pruning or bucket-aligned joins. Writers
must not write to a table whose layout contains a transform they do not recognize.

## Partition values

Every transform maps a source value to a _partition value_:

- Each transform spec defines the type of its partition column and the range of its
  partition values. A partition column that the layout describes must have the type
  that its transform defines.
- A null source value maps to a null partition value.


## Nesting

Each transform defines a set of _family_ members, which consist of every member except
the one that sets `n_buckets`. Two partition fields are _nesting-compatible_ if
their transforms have the same `"@type"` and equal family members. For
nesting-compatible fields with `n_buckets` values `m <= n`, every source value
satisfies:

```
bucket_m = bucket_n % m
```

Equivalently, `bucket_m = bucket_n & (m - 1)`. Rows from bucket `b` of the
`n`-bucket table therefore can only match rows from bucket `b % m` of the `m`-bucket
table, which allows bucket-aligned joins between tables with different bucket counts.

Partition fields with different transform `"@type"` values are never
nesting-compatible.

## Example

A synapse table bucketed on the presynaptic root ID into 64 buckets:

```json
{
  "@type": "cave_layout_v1",
  "transforms": [
    {
      "name": "pre_pt_root_id_bucket",
      "source": "pre_pt_root_id",
      "transform": { "@type": "iceberg_bucket", "n_buckets": 64 }
    }
  ]
}
```

## Open questions

- `VACUUM` in delta-rs leaves `_info/` in place (see `tests/test_layout.py`). Confirm
  the same for Spark. Both treat `_`-prefixed paths as hidden, but with exceptions
  for partition directories.
- The "Bucket values" and "Nesting" sections apply to every transform. A spatial
  curve, such as a Morton or Hilbert code, nests by a right shift, not by `% m`. It
  can have a different count per axis and need more than 31 bits. Will these rules
  move into each transform spec?
- Is an empty `"transforms"` array allowed? This spec says that it must be
  non-empty, but the Python implementation does not check.
- Must a reader ignore a transform with an unknown `"@type"`, as this spec says, or
  reject it, as the Python implementation does now?
- A Delta `domainMetadata` domain with a custom name can keep the layout in the Delta
  log, in the same transaction as the data. Python `deltalake` cannot read or write
  domain metadata yet. When it can, will the layout move there?
- `layout.json` is not written in the same transaction as the Delta log, so the two
  can disagree. Should the layout record the Delta table version at which it took
  effect?
- Should a default partition column name, such as `<source>_bucket`, be recommended?
- `"name"` and `"source"` refer to columns by name, so renaming either column makes
  the layout stale. Iceberg avoids this by referring to source columns by ID. Should
  the layout use Delta column mapping IDs (`delta.columnMapping.id`), which requires
  column mapping to be enabled, or should renaming a referenced column require
  rewriting the layout?
