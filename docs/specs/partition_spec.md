# Partition spec

> Status: draft.

This spec adds a small piece of metadata to an ordinary Delta table. It is not a new
table format. The table stays a standard Delta table that any Delta reader can read.

The metadata is for a Delta table whose rows are partitioned by deterministic
transforms of one or more source columns, for example a hash bucket of a segment ID.
Delta records which columns a table is partitioned by, but not how those partition
values were computed. The _partition spec_ records that information in the table
metadata, so that any reader can compute the partition value for a given source
value and use it for file pruning or partition-wise joins.

The partition spec is table-level: every file in the table shares one partition spec.

## Storage

The partition spec is stored as a JSON string in the Delta table `configuration`,
under the key `cave.partition_spec`:

```json
"configuration": {
  "cave.partition_spec": "{\"@type\":\"cave_partition_spec_v1\",\"fields\":[...]}"
}
```

- The key is the single source of truth for the partition spec. The partition spec
  must not also be stored in a sidecar file.
- The `configuration` is part of the Delta table metadata, so the partition spec
  commits in the same transaction as the data. A reader of a table version gets the
  partition spec of that version.
- A Delta table without the key has no cavebear partition spec.
- An overwrite that changes the partition columns keeps the key. A reader must check
  that each field names a partition column of the table version that it reads, with
  the type that its transform defines.
- Delta writers that check `configuration` keys must accept this key. In Python
  `deltalake`, `DeltaTable.create(..., raise_if_key_not_exists=False)` does this.

## Partition spec format

The partition spec is a JSON object with the following members:

- `"@type"`: Must be `"cave_partition_spec_v1"`.
- `"fields"`: A non-empty array of _partition field_ objects, described below.
  Each entry describes one transformed Delta partition column, and no two entries may
  share the same `"name"`. The order of entries has no meaning; Delta's
  `partitionColumns` defines the order of partition columns. Partition columns not
  named by any entry are ordinary Delta partition columns, and the partition spec
  does not describe them.

Each partition field object has the following members:

- `"name"`: String. The name of the Delta partition column that stores the partition
  value. It must appear in the Delta table's `partitionColumns`. The recommended
  name is `<source>_partition`, for example `pre_pt_root_id_partition`. The name
  does not depend on the transform, so a table keeps its column names if its
  transform changes. Two fields on one source need different names. A reader must
  use the `"name"` that is stored, and must not derive it from `"source"`.
- `"source"`: String. The name of the column the partition value is computed from. It
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

Readers must reject a partition spec whose top-level `"@type"` they do not recognize.
As in Iceberg, a reader that does not recognize a transform's `"@type"` must ignore
that partition field, and must not use it for file pruning or partition-wise joins.
The reads still return correct rows, but open more files. A reader should warn the
user when this happens. Writers must not write to a table whose partition spec
contains a transform they do not recognize.

## Partition values

Every transform maps a source value to a _partition value_:

- Each transform spec defines the type of its partition column and the range of its
  partition values. A partition column that the partition spec describes must have
  the type that its transform defines.
- A null source value maps to a null partition value.


## Nesting

Each transform defines a set of _family_ members, which consist of every member except
the one that sets the partition count. Two partition fields _nest_ if their
transforms have the same `"@type"` and equal family members. For fields that nest,
with partition counts `m <= n`, every source value satisfies:

```
partition_m = partition_n % m
```

Equivalently, `partition_m = partition_n & (m - 1)`. Partition `b` of the field with
`m` partitions holds the rows of the partitions `b, b + m, b + 2m, ...` of the field
with `n` partitions: these partitions are _nested_ in partition `b`. Rows from
partition `p` of the `n`-partition table therefore can only match rows from partition
`p % m` of the `m`-partition table. This allows partition-wise joins between tables
with different partition counts: a join that runs one partition at a time.

Partition fields with different transform `"@type"` values never nest.

## Example

A synapse table partitioned on the presynaptic root ID with `iceberg_bucket` into 64
partitions:

```json
{
  "@type": "cave_partition_spec_v1",
  "fields": [
    {
      "name": "pre_pt_root_id_partition",
      "source": "pre_pt_root_id",
      "transform": { "@type": "iceberg_bucket", "n_buckets": 64 }
    }
  ]
}
```

## Open questions

- The "Nesting" section applies to every transform. A spatial curve, such as a Morton
  or Hilbert code, nests by a right shift, not by `% m`, and it can have a different
  count per axis. Will the nesting rule move into each transform spec?
- A Delta `domainMetadata` domain with a custom name can keep the partition spec in
  the Delta log as its own action. Python `deltalake` cannot read or write domain
  metadata yet. When it can, will the partition spec move there?
- `"name"` and `"source"` refer to columns by name, so renaming either column makes
  the partition spec stale. Iceberg avoids this by referring to source columns by ID.
  Should the partition spec use Delta column mapping IDs (`delta.columnMapping.id`),
  which requires column mapping to be enabled, or should renaming a referenced column
  require rewriting the partition spec?
