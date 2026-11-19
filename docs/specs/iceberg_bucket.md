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

## Partition column

The partition column has type `int32`. Its values are in the range `[0, n_buckets)`,
and a null source value gives a null. Reasons:

- Iceberg defines the result type of `bucket[N]` as `int`, a 32-bit signed integer.
- `n_buckets` is at most `2**30`, so every bucket number fits.

## Test vectors

Each row gives a value, the signed 32-bit hash, and the bucket for three values of
`n_buckets`. For a negative value, the row also gives its `uint64` view. A `uint64`
value and its `int64` view have the same bucket.

| value                  | `uint64` view        | `murmur3_x86_32` | 16  | 64  | 1024 |
| ---------------------- | -------------------- | ---------------- | --- | --- | ---- |
| `0`                    |                      | `1669671676`     | 12  | 60  | 764  |
| `1`                    |                      | `1392991556`     | 4   | 4   | 324  |
| `34`                   |                      | `2017239379`     | 3   | 19  | 339  |
| `-34`                  | `2**64 - 34`         | `-2117612851`    | 13  | 13  | 717  |
| `-1`                   | `2**64 - 1`          | `1651860712`     | 8   | 40  | 232  |
| `2**63 - 1`            |                      | `-2106506049`    | 15  | 63  | 191  |
| `-2**63`               | `2**63`              | `1366273829`     | 5   | 37  | 805  |
| `-2401053088876216593` | `0xDEADBEEFDEADBEEF` | `-1017682423`    | 9   | 9   | 521  |

The hash of `34` matches Iceberg spec Appendix B.

These implementations of the Iceberg bucket transform have test vectors that the
table can be checked against:

- Iceberg Java: `api/src/test/java/org/apache/iceberg/transforms/TestBucketing.java`
- pyiceberg: `tests/test_transforms.py`
- iceberg-rust: `crates/iceberg/src/transform/bucket.rs`
- iceberg-cpp: `src/iceberg/test/bucket_util_test.cc`, with a `uint64` value
  `>= 2**63`
- Trino: `plugin/trino-iceberg/src/test/java/io/trino/plugin/iceberg/TestIcebergBucketing.java`,
  with `Long.MIN_VALUE` and `Long.MAX_VALUE`

`pyiceberg_core.transform.bucket` computes the transform on an Arrow array of
`int32` or `int64` values. It rejects `uint64` arrays, so reinterpret a `uint64`
array as `int64` before the call.

## Relation to Iceberg

A cavebear partition field using this transform corresponds to the Iceberg partition
field:

```json
{ "source-id": <id of source>, "name": "<column>", "transform": "bucket[<n_buckets>]" }
```

Iceberg allows any positive `N`. Cavebear requires a power of two so that buckets nest
across tables.
