import json

import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake

from cavebear import LAYOUT_PATH, IcebergBucket, Layout, PartitionField
from cavebear._store import _split_options, open_store

#: The layout that neuron-hks-analysis writes for its bucketed synapse table.
SYNAPSE_LAYOUT = {
    "@type": "cave_layout_v1",
    "transforms": [
        {
            "name": "post_pt_root_id_bucket",
            "source": "post_pt_root_id",
            "transform": {"@type": "iceberg_bucket", "n_buckets": 4},
        }
    ],
}

LAYOUT = Layout((PartitionField("post_pt_root_id_bucket", "post_pt_root_id", IcebergBucket(4)),))


def test_json_round_trip():
    text = json.dumps(SYNAPSE_LAYOUT, indent=2) + "\n"
    assert Layout.from_json(text) == LAYOUT
    assert LAYOUT.to_json() == text


def test_rejects_bad_layouts():
    with pytest.raises(ValueError, match="@type"):
        Layout.from_dict({**SYNAPSE_LAYOUT, "@type": "cave_layout_v2"})
    with pytest.raises(ValueError, match="list"):
        Layout.from_dict({"@type": "cave_layout_v1", "transforms": {}})
    with pytest.raises(ValueError, match="unique"):
        Layout(LAYOUT.fields * 2)
    unknown = {**SYNAPSE_LAYOUT["transforms"][0], "transform": {"@type": "morton", "n": 8}}
    with pytest.raises(ValueError, match="unknown transform"):
        Layout.from_dict({"@type": "cave_layout_v1", "transforms": [unknown]})


def test_write_and_read(tmp_path):
    root = tmp_path / "table"
    LAYOUT.write(root)
    assert json.loads((root / LAYOUT_PATH).read_text()) == SYNAPSE_LAYOUT
    assert Layout.read(root) == LAYOUT
    assert Layout.read(root.as_uri()) == LAYOUT


def test_write_is_create_only(tmp_path):
    other = Layout((PartitionField("b", "a", IcebergBucket(8)),))
    LAYOUT.write(tmp_path)
    with pytest.raises(FileExistsError):
        other.write(tmp_path)
    assert Layout.read(tmp_path) == LAYOUT
    other.write(tmp_path, overwrite=True)
    assert Layout.read(tmp_path) == other


def test_read_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Layout.read(tmp_path)
    with pytest.raises(FileNotFoundError):
        Layout.read(tmp_path / "nope")


def test_split_options():
    options = {
        "AWS_REGION": "us-east-1",
        "max_retries": 3,
        "retry_timeout_ms": 100,
        "AWS_S3_LOCKING_PROVIDER": "dynamodb",
        "allow_http": "true",
        "TIMEOUT": "30s",
    }
    config, client_options = _split_options(options)
    assert config == {"AWS_REGION": "us-east-1"}
    assert client_options == {"allow_http": "true", "timeout": "30s"}
    open_store("s3://bucket/table", options)


def test_unknown_option_raises():
    with pytest.raises(Exception, match="not_a_key"):
        open_store("s3://bucket/table", {"not_a_key": "x"})


def test_vacuum_keeps_layout(tmp_path):
    data = pa.table({"id": [1, 2, 3], "id_bucket": pa.array([0, 1, 0], pa.int32())})
    write_deltalake(tmp_path, data, partition_by=["id_bucket"])
    write_deltalake(tmp_path, data, partition_by=["id_bucket"], mode="overwrite")
    LAYOUT.write(tmp_path)
    removed = DeltaTable(tmp_path).vacuum(
        retention_hours=0, enforce_retention_duration=False, dry_run=False
    )
    assert removed
    assert Layout.read(tmp_path) == LAYOUT
