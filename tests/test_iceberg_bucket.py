import numpy as np
import pyarrow as pa
import pytest
from pyiceberg_core import transform as iceberg

from cavebear import IcebergBucket
from cavebear.transforms import UnknownTransform, parse_transform

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


def test_rejects_unknown_members():
    with pytest.raises(ValueError):
        parse_transform({"@type": "iceberg_bucket", "n_buckets": 64, "seed": 1})


def test_keeps_unknown_types():
    raw = {"@type": "neuroglancer_uint64_sharded_v1", "shard_bits": 4}
    with pytest.warns(UserWarning, match="neuroglancer_uint64_sharded_v1"):
        assert parse_transform(raw) == UnknownTransform(raw)


@pytest.mark.parametrize("n", [1, 16, 1024, 2**30])
def test_array_matches_pyiceberg_core_and_scalar(n):
    rng = np.random.default_rng(n)
    values = pa.array(rng.integers(-(2**63), 2**63, size=10_000, dtype=np.int64))
    got = IcebergBucket(n).apply(values)
    assert got.type == pa.int32()
    assert got.to_pylist() == iceberg.bucket(values, n).to_pylist()
    assert got.to_pylist()[:500] == [IcebergBucket(n).bucket(v) for v in values.to_pylist()[:500]]


def test_array_vectors():
    signed = pa.array([v for v, *_ in VECTORS], pa.int64())
    unsigned = signed.view(pa.uint64())
    for n in (16, 64, 1024):
        expected = [buckets[n] for *_, buckets in VECTORS]
        assert IcebergBucket(n).apply(signed).to_pylist() == expected
        assert IcebergBucket(n).apply(unsigned).to_pylist() == expected


@pytest.mark.parametrize("kind", [pa.int8(), pa.int16(), pa.int32(), pa.uint8(), pa.uint16(), pa.uint32()])
def test_array_small_integer_types(kind):
    values = pa.array([0, 1, 34, 100], kind)
    expected = [IcebergBucket(64).bucket(v) for v in values.to_pylist()]
    assert IcebergBucket(64).apply(values).to_pylist() == expected


def test_array_nulls_slices_and_chunks():
    values = pa.array([34, None, 0, 1], pa.int64())
    assert IcebergBucket(16).apply(values).to_pylist() == [3, None, 12, 4]
    assert IcebergBucket(16).apply(values.slice(2)).to_pylist() == [12, 4]
    chunked = pa.chunked_array([values.slice(0, 2), values.slice(2)])
    got = IcebergBucket(16).apply(chunked)
    assert isinstance(got, pa.ChunkedArray) and got.to_pylist() == [3, None, 12, 4]


def test_array_rejects_non_integers():
    with pytest.raises(TypeError):
        IcebergBucket(16).apply(pa.array([1.0]))


@pytest.mark.parametrize(
    "n, expected",
    [(-5, 1), (0, 1), (1, 1), (2, 2), (3, 4), (64, 64), (65, 128), (2**30 - 1, 2**30), (2**30, 2**30)],
)
def test_round_up(n, expected):
    assert IcebergBucket.round_up(n) == expected
    IcebergBucket(expected)  # always a valid count


def test_round_up_rejects_too_large():
    with pytest.raises(ValueError, match="2\\*\\*30"):
        IcebergBucket.round_up(2**30 + 1)
