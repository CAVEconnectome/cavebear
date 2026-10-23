# Iceberg bucket transform

> Status: draft.

The `iceberg_bucket` transform assigns a row to a bucket by hashing an integer source
value, using the `bucket[N]` transform defined by the
[Apache Iceberg table spec](https://iceberg.apache.org/spec/#bucket-transform-details).
For the same source values and the same `n_buckets`, the bucket numbers are
identical to those of an Iceberg table partitioned by `bucket[n_buckets]` on that
column.

## Transform specification

The transform object has the following members:

- `"@type"`: Must be `"iceberg_bucket"`.
- `"n_buckets"`: Integer. The number of buckets. Must be a power of two in the range
  `[1, 2**30]`.

There are no family members (see [layout.md](./layout.md#nesting)), so any two
`iceberg_bucket` partition fields are nesting-compatible.

## Source types

The source column must have an integer type. Before hashing, each value is converted
to a 64-bit two's complement integer:

- Signed integers (`int8`, `int16`, `int32`, `int64`) and unsigned integers up to
  `uint32` are converted by value.
- `uint64` values are reinterpreted bit for bit as `int64`. Values `>= 2**63`
  therefore become negative, but the 8 bytes being hashed are unchanged.

Iceberg hashes `int` and `long` values the same way, so `int32` and `int64` columns
holding the same values produce the same buckets.

## Computing the bucket number

```
h = murmur3_x86_32(int64_le_bytes(value), seed=0)   # signed 32-bit result
bucket = (h & 0x7FFFFFFF) % n_buckets
```

where `int64_le_bytes` is the 8-byte little-endian encoding of the converted value.
Because `n_buckets` is a power of two, `bucket = h & (n_buckets - 1)`.

## Test vectors

| value | `murmur3_x86_32` | `n_buckets` | bucket |
| ----- | ---------------- | ------------- | ------ |
| `34`  | `2017239379`     | `16`          | `3`    |

The hash value matches Iceberg spec Appendix B.

TODO: add vectors for a `uint64` value `>= 2**63` and for several `n_buckets`.

## Relation to Iceberg

A cistern partition field using this transform corresponds to the Iceberg partition
field:

```json
{ "source-id": <id of source>, "name": "<column>", "transform": "bucket[<n_buckets>]" }
```

Iceberg allows any positive `N`. Cistern requires a power of two so that buckets nest
across tables.
