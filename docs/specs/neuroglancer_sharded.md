# Neuroglancer sharded transform

> Status: draft.

The `neuroglancer_uint64_sharded_v1` transform assigns a row to the shard that
Neuroglancer's
[precomputed sharded format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md)
would assign to its source ID. A table using this transform lines up with an
existing precomputed dataset keyed by the same IDs, such as segment meshes or
skeletons: bucket `b` of the table holds the rows whose IDs are stored in shard file
`b` of the dataset.

## Transform specification

The transform object is a Neuroglancer sharding specification, with the following
members:

- `"@type"`: Must be `"neuroglancer_uint64_sharded_v1"`.
- `"preshift_bits"`: Integer. The number of low-order bits of the ID that do not
  contribute to the hashed ID.
- `"hash"`: The hash function applied to the shifted ID. Must be one of:
  - `"identity"`: The identity function.
  - `"murmurhash3_x86_128"`: MurmurHash3_x86_128 with seed 0, applied to the shifted
    ID in little-endian encoding. The low 8 bytes of the resulting hash code are
    treated as a little-endian `uint64`.
- `"minishard_bits"`: Integer. The number of low-order bits of the hashed ID that
  select the minishard. These bits do not affect the bucket number.
- `"shard_bits"`: Integer in the range `[0, 31]`. The number of bits of the hashed ID
  that select the shard. `num_buckets` is `2**shard_bits`.
- `"minishard_index_encoding"`, `"data_encoding"`: Optional, and ignored. They are
  permitted so that the `"sharding"` object of an existing precomputed `info` file can
  be copied verbatim.

`preshift_bits + minishard_bits + shard_bits` must not exceed 64.

The family members (see [layout.md](./layout.md#nesting)) are `"hash"`,
`"preshift_bits"` and `"minishard_bits"`. Two partition fields using this transform
are nesting-compatible only if all three are equal; only `"shard_bits"` may differ.

## Source types

The source column must have an integer type. Before hashing, each value is converted
to `uint64`:

- Unsigned integers are converted by value.
- Signed integers are converted to `int64` by value, then reinterpreted bit for bit as
  `uint64`.

## Computing the bucket number

```
hashed = hash(value >> preshift_bits)
bucket = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
```

The bucket number equals the Neuroglancer shard number: bits
`[minishard_bits, minishard_bits + shard_bits)` of the hashed ID. The minishard number
is not stored.

## Test vectors

TODO: generate vectors for both `"hash"` values and check them against Neuroglancer
or cloud-volume.

## Relation to Neuroglancer

For a precomputed dataset with the same sharding specification, the rows in bucket
`b` correspond to the shard file `<b>.shard`, where `<b>` is the lowercase base-16
shard number zero-padded to `ceil(shard_bits / 4)` digits.
