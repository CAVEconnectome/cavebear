import json

import pytest

from cavebear import CONFIG_KEY, IcebergBucket, PartitionField, PartitionSpec, UnknownTransform

#: The partition spec of a synapse table bucketed on the postsynaptic root ID.
SYNAPSE_SPEC = {
    "@type": "cave_partition_spec_v1",
    "fields": [
        {
            "name": "post_pt_root_id_bucket",
            "source": "post_pt_root_id",
            "transform": {"@type": "iceberg_bucket", "n_buckets": 4},
        }
    ],
}

SPEC = PartitionSpec(
    (PartitionField("post_pt_root_id_bucket", "post_pt_root_id", IcebergBucket(4)),)
)


def test_json_round_trip():
    assert PartitionSpec.from_dict(SYNAPSE_SPEC) == SPEC
    assert json.loads(SPEC.to_json()) == SYNAPSE_SPEC
    assert PartitionSpec.from_json(SPEC.to_json()) == SPEC


def test_configuration_round_trip():
    configuration = SPEC.to_configuration()
    assert list(configuration) == [CONFIG_KEY] == ["cave.partition_spec"]
    assert PartitionSpec.from_configuration({**configuration, "delta.appendOnly": "true"}) == SPEC
    assert PartitionSpec.from_configuration({}) is None


def test_rejects_bad_specs():
    with pytest.raises(ValueError, match="@type"):
        PartitionSpec.from_dict({**SYNAPSE_SPEC, "@type": "cave_layout_v1"})
    with pytest.raises(ValueError, match="list"):
        PartitionSpec.from_dict({**SYNAPSE_SPEC, "fields": {}})
    with pytest.raises(ValueError, match="at least one"):
        PartitionSpec.from_dict({**SYNAPSE_SPEC, "fields": []})
    with pytest.raises(ValueError, match="members"):
        PartitionSpec.from_dict({**SYNAPSE_SPEC, "transforms": []})
    with pytest.raises(ValueError, match="unique"):
        PartitionSpec(SPEC.fields * 2)


def test_unknown_transform_warns_and_round_trips():
    raw = {"@type": "morton", "bits": [4, 4, 2]}
    data = {**SYNAPSE_SPEC, "fields": [*SYNAPSE_SPEC["fields"], {"name": "cell", "source": "x", "transform": raw}]}
    with pytest.warns(UserWarning, match="'morton'"):
        spec = PartitionSpec.from_dict(data)
    assert spec.fields[1].transform == UnknownTransform(raw)
    assert not spec.fields[1].known
    assert json.loads(spec.to_json()) == data
    with pytest.raises(ValueError, match="unknown transforms"):
        spec.require_known()
    SPEC.require_known()
