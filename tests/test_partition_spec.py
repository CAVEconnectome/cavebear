import json

import pytest

from cavebear import CONFIG_KEY, IcebergBucket, PartitionField, PartitionSpec, UnknownTransform

#: The partition spec of a synapse table bucketed on the postsynaptic root ID.
SYNAPSE_SPEC = {
    "@type": "cave_partition_spec_v1",
    "fields": [
        {
            "name": "post_pt_root_id_partition",
            "source": "post_pt_root_id",
            "transform": {"@type": "iceberg_bucket", "n_buckets": 4},
        }
    ],
}

SPEC = PartitionSpec(
    (PartitionField(source="post_pt_root_id", transform=IcebergBucket(4), name="post_pt_root_id_partition"),)
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


def test_default_name():
    field = PartitionField(source="post_pt_root_id", transform=IcebergBucket(4))
    assert field.name == "post_pt_root_id_partition"
    assert field == SPEC.fields[0]
    named = PartitionField(source="post_pt_root_id", transform=IcebergBucket(4), name="b")
    assert named.name == "b"


def test_field_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        PartitionField("post_pt_root_id_partition", "post_pt_root_id", IcebergBucket(4))


def test_mapping_spec_equals_full_form():
    spec = PartitionSpec({"post_pt_root_id": IcebergBucket(4)})
    assert spec == SPEC
    assert spec.to_dict() == SPEC.to_dict() == SYNAPSE_SPEC


def test_mapping_spec_keeps_order():
    spec = PartitionSpec({"pre_pt_root_id": IcebergBucket(8), "post_pt_root_id": IcebergBucket(4)})
    assert [(f.name, f.source, f.transform.n_buckets) for f in spec.fields] == [
        ("pre_pt_root_id_partition", "pre_pt_root_id", 8),
        ("post_pt_root_id_partition", "post_pt_root_id", 4),
    ]
    with pytest.raises(ValueError, match="at least one"):
        PartitionSpec({})


def test_two_fields_on_one_source_need_names():
    def field(n, **kwargs):
        return PartitionField(source="post_pt_root_id", transform=IcebergBucket(n), **kwargs)

    with pytest.raises(ValueError, match="unique"):
        PartitionSpec((field(4), field(16)))
    spec = PartitionSpec((field(4), field(16, name="post_pt_root_id_16")))
    assert [f.transform.n_buckets for f in spec.fields_for("post_pt_root_id")] == [4, 16]
