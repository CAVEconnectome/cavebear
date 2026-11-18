import numpy as np
import pyarrow as pa
import pytest
from pyiceberg_core import transform as iceberg

from cavebear import IcebergBucket
from cavebear.transforms import parse_transform

#: The test vectors in docs/specs/iceberg_bucket.md:
#: (int64 value, uint64 view or None, murmur3_x86_32, {n_buckets: bucket}).
VECTORS = [
    (0, None, 1669671676, {16: 12, 64: 60, 1024: 764}),
    (1, None, 1392991556, {16: 4, 64: 4, 1024: 324}),
    (34, None, 2017239379, {16: 3, 64: 19, 1024: 339}),
    (-34, 2**64 - 34, -2117612851, {16: 13, 64: 13, 1024: 717}),
    (-1, 2**64 - 1, 1651860712, {16: 8, 64: 40, 1024: 232}),
    (2**63 - 1, None, -2106506049, {16: 15, 64: 63, 1024: 191}),
    (-(2**63), 2**63, 1366273829, {16: 5, 64: 37, 1024: 805}),
    (-2401053088876216593, 0xDEADBEEFDEADBEEF, -1017682423, {16: 9, 64: 9, 1024: 521}),
]


@pytest.mark.parametrize("value, unsigned, hash_value, buckets", VECTORS)
def test_vectors(value, unsigned, hash_value, buckets):
    # bucket[2**30] keeps the 30 low bits of the hash with the sign bit cleared.
    assert IcebergBucket(2**30).bucket(value) == hash_value & (2**30 - 1)
    for n, expected in buckets.items():
        assert IcebergBucket(n).bucket(value) == expected
        if unsigned is not None:
            assert IcebergBucket(n).bucket(unsigned) == expected


@pytest.mark.parametrize("n", [1, 16, 1024, 2**30])
def test_matches_pyiceberg_core(n):
    rng = np.random.default_rng(n)
    values = rng.integers(-(2**63), 2**63, size=10_000, dtype=np.int64)
    expected = iceberg.bucket(pa.array(values), n).to_pylist()
    assert [IcebergBucket(n).bucket(int(v)) for v in values] == expected


def test_buckets_nest():
    rng = np.random.default_rng(0)
    values = [int(v) for v in rng.integers(-(2**63), 2**63, size=1000, dtype=np.int64)]
    for log_n in range(11):
        fine = IcebergBucket(2**log_n)
        for log_m in range(log_n + 1):
            coarse = IcebergBucket(2**log_m)
            assert all(fine.bucket(v) % coarse.n_buckets == coarse.bucket(v) for v in values)


@pytest.mark.parametrize("n", [0, 3, -4, 2**31, 2.0, True])
def test_rejects_bad_n_buckets(n):
    with pytest.raises(ValueError):
        IcebergBucket(n)


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1])
def test_rejects_out_of_range_values(value):
    with pytest.raises(ValueError):
        IcebergBucket(16).bucket(value)


def test_dict_round_trip():
    data = {"@type": "iceberg_bucket", "n_buckets": 64}
    assert parse_transform(data) == IcebergBucket(64)
    assert IcebergBucket(64).to_dict() == data


def test_rejects_unknown_members_and_types():
    with pytest.raises(ValueError):
        parse_transform({"@type": "iceberg_bucket", "n_buckets": 64, "seed": 1})
    with pytest.raises(ValueError):
        parse_transform({"@type": "neuroglancer_uint64_sharded_v1", "shard_bits": 4})
