"""Create, write and read Delta tables that have a partition spec."""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .partition_spec import CONFIG_KEY, PartitionSpec
from .transforms import IcebergBucket, to_arrow_int64

def _arrow_schema(dt: DeltaTable) -> pa.Schema:
    return pa.schema(dt.schema().to_arrow())


def partition_spec(dt: DeltaTable) -> PartitionSpec:
    """The partition spec of the snapshot ``dt``.

    Parameters
    ----------
    dt
        The Delta table. The spec comes from the configuration of its current
        version.

    Returns
    -------
    spec :
        The partition spec of ``dt``.

    Raises
    ------
    ValueError
        If the table has no spec, or if the spec does not describe the table
        (``PartitionSpec.validate_table``).
    """
    metadata = dt.metadata()
    spec = PartitionSpec.from_configuration(metadata.configuration)
    if spec is None:
        raise ValueError(f"{dt.table_uri} has no cavebear partition spec ({CONFIG_KEY!r} is not set)")
    spec.validate_table(_arrow_schema(dt), metadata.partition_columns)
    return spec


def as_partition_spec(table: DeltaTable | PartitionSpec) -> PartitionSpec:
    """``table`` itself if it is a partition spec, or else its partition spec.

    Parameters
    ----------
    table
        A Delta table or a partition spec.

    Returns
    -------
    spec :
        The partition spec.
    """
    return table if isinstance(table, PartitionSpec) else partition_spec(table)


def partition_values(
    table: DeltaTable | PartitionSpec, where: Mapping[str, Sequence[int]]
) -> dict[str, list[int]]:
    """The partition values that can hold the rows of a filter.

    A filter on these values only selects files. The reader must also filter the
    source column itself, because one partition holds many source values.

    Parameters
    ----------
    table
        A Delta table or its partition spec.
    where
        Maps a source column to the values that a read keeps.

    Returns
    -------
    values :
        Maps each partition column computed from the columns of ``where`` to the
        sorted partition values of those values.
    """
    spec = as_partition_spec(table)
    out: dict[str, list[int]] = {}
    for source, values in where.items():
        keys = to_arrow_int64(values)
        for f in spec.fields_for(source):
            found = set(f.transform.apply(keys).to_pylist()) - {None}
            out[f.name] = sorted(found & set(out.get(f.name, found)))
    return out


def create_delta(
    uri: str | Path,
    schema: Any,
    spec: PartitionSpec,
    *,
    partition_by: Sequence[str] | None = None,
    storage_options: dict[str, str] | None = None,
) -> DeltaTable:
    """Create an empty Delta table with the partition spec ``spec``.

    The partition columns are added to the schema with the type that each transform
    defines. The spec is stored in the table configuration.

    Parameters
    ----------
    uri
        The location of the new table.
    schema
        The schema of the data that you will write, without the partition columns of
        the spec. Anything that ``pyarrow.schema`` accepts.
    spec
        The partition spec of the table.
    partition_by
        The order of the partition columns. It must name every field of the spec,
        and it can add ordinary partition columns. The default is the spec fields,
        in order.
    storage_options
        Options for the object store, as in ``deltalake``.

    Returns
    -------
    table :
        The new, empty table.

    Raises
    ------
    ValueError
        If the spec does not describe the table, for example if ``partition_by``
        leaves out a spec field.
    """
    schema = spec.add_partition_columns(pa.schema(schema))
    partition_by = list(partition_by) if partition_by is not None else [f.name for f in spec.fields]
    # Check the table before it exists, with the same check as a reader.
    spec.validate_table(schema, partition_by)
    return DeltaTable.create(
        str(uri),
        schema,
        partition_by=partition_by,
        configuration=spec.to_configuration(),
        storage_options=storage_options,
        raise_if_key_not_exists=False,
    )


def _with_partitions(
    reader: pa.RecordBatchReader, spec: PartitionSpec, target: pa.Schema
) -> pa.RecordBatchReader:
    schema = spec.add_partition_columns(reader.schema)
    if set(schema.names) == set(target.names):
        schema = pa.schema([schema.field(name) for name in target.names])

    def batches() -> Iterator[pa.RecordBatch]:
        for batch in reader:
            columns = dict(zip(batch.schema.names, batch.columns))
            for f in spec.fields:
                columns[f.name] = f.transform.apply(columns[f.source])
            yield pa.RecordBatch.from_arrays([columns[n] for n in schema.names], schema=schema)

    return pa.RecordBatchReader.from_batches(schema, batches())


def write_delta(
    table: DeltaTable | str | Path,
    data: Any,
    *,
    mode: Literal["append", "overwrite"] = "append",
    storage_options: dict[str, str] | None = None,
) -> None:
    """Write ``data`` to a Delta table with a partition spec.

    ``write_delta`` computes the partition columns of the spec from the source
    columns, batch by batch. As with ``write_deltalake``, a ``DeltaTable`` argument
    moves to the new version.

    Parameters
    ----------
    table
        The table, or its location.
    data
        Any object that exports an Arrow stream (``__arrow_c_stream__``), such as a
        pyarrow table, a Polars DataFrame or a DuckDB relation. Collect a Polars
        LazyFrame first. The data must not have the partition columns of the spec.
    mode
        Append to the table, or overwrite it.
    storage_options
        Options for the object store, as in ``deltalake``. Used only if ``table`` is
        a location.

    Raises
    ------
    ValueError
        If the table has no partition spec, or if the spec does not describe the
        table.
    """
    if not isinstance(table, DeltaTable):
        table = DeltaTable(str(table), storage_options=storage_options)
    spec = partition_spec(table)
    reader = _with_partitions(pa.RecordBatchReader.from_stream(data), spec, _arrow_schema(table))
    write_deltalake(table, reader, mode=mode)


def nested_partitions(value: int, coarse_n: int, fine_n: int) -> list[int]:
    """The partitions of ``bucket[fine_n]`` that are nested in a partition of
    ``bucket[coarse_n]``.

    Both counts are powers of two and ``coarse_n <= fine_n``, so the result is every
    ``v`` in ``[0, fine_n)`` with ``v % coarse_n == value``.

    Parameters
    ----------
    value
        A partition value of ``bucket[coarse_n]``.
    coarse_n
        The smaller bucket count.
    fine_n
        The larger bucket count.

    Returns
    -------
    partitions :
        The partition values of ``bucket[fine_n]``, in increasing order.

    Raises
    ------
    ValueError
        If a count is not a power of two, if ``coarse_n > fine_n``, or if ``value``
        is not in ``[0, coarse_n)``.

    Examples
    --------
    >>> nested_partitions(5, 64, 128)
    [5, 69]
    """
    IcebergBucket(coarse_n), IcebergBucket(fine_n)  # both must be powers of two
    if coarse_n > fine_n:
        raise ValueError(f"bucket[{fine_n}] is not nested in bucket[{coarse_n}]")
    if not 0 <= value < coarse_n:
        raise ValueError(f"partition value {value} is not in [0, {coarse_n})")
    return list(range(value, fine_n, coarse_n))


def _nested_filter(
    dt: DeltaTable, spec: PartitionSpec, source: str, value: int, coarse_n: int
) -> dict[str, list[int]]:
    """For each ``iceberg_bucket`` field on ``source``, its partitions nested in
    partition ``value`` of ``bucket[coarse_n]``."""
    # NOTE: partition-wise joins support only iceberg_bucket fields, here and in the
    # default ``n`` of ``partitionwise_join_values``. To add another transform, such as
    # neuroglancer_uint64_sharded_v1 (nested by ``shard_bits`` when ``hash``,
    # ``preshift_bits`` and ``minishard_bits`` are equal), give each transform its
    # own partition count and nesting rule, and check that all fields of a join have
    # the same @type and equal family members (docs/specs/partition_spec.md, "Nesting").
    fields = [f for f in spec.fields_for(source) if isinstance(f.transform, IcebergBucket)]
    if not fields:
        raise ValueError(f"{dt.table_uri} has no iceberg_bucket field on {source!r}")
    return {f.name: nested_partitions(value, coarse_n, f.transform.n_buckets) for f in fields}


def partitionwise_join_values(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[dict[str, list[int]]]]]:
    """The partition filters of a partition-wise join, one partition at a time.

    The join runs over the partitions of ``bucket[n]``. For each partition value,
    this gives one filter per table: its partitions nested in that partition. A join
    of the tables, each filtered this way, is complete for the source values of that
    partition.

    Parameters
    ----------
    tables
        The tables to join. Each must have an ``iceberg_bucket`` field on its source
        column of the join.
    on
        The source column of the join in each table, or one name for all of them.
    n
        The partition count of the steps. The default, and the largest allowed
        value, is the smallest partition count of the tables. A smaller ``n`` gives
        fewer, larger steps. For example, tables with 256 partitions and ``n=32``
        read 8 partitions of each table per step, as ``BUCKETS_PER_BATCH`` does in
        neuron-hks-analysis.

    Yields
    ------
    value :
        The partition value of the step, in ``[0, n)``.
    filters :
        One filter per table, in the order of ``tables``. Each maps a partition
        column to the partition values of the step.

    Raises
    ------
    ValueError
        If ``on`` does not give one column per table, or if a table has no
        ``iceberg_bucket`` field on its source column. The check runs before the
        first step.
    """
    sources = [on] * len(tables) if isinstance(on, str) else list(on)
    if len(sources) != len(tables):
        raise ValueError("on needs one source column per table")
    specs = [partition_spec(dt) for dt in tables]
    if n is None:
        n = min(
            f.transform.n_buckets
            for spec, source in zip(specs, sources)
            for f in spec.fields_for(source)
            if isinstance(f.transform, IcebergBucket)
        )
    # Check every table before the first partition.
    for dt, spec, source in zip(tables, specs, sources):
        _nested_filter(dt, spec, source, 0, n)
    for value in range(n):
        yield value, [
            _nested_filter(dt, spec, source, value, n)
            for dt, spec, source in zip(tables, specs, sources)
        ]
