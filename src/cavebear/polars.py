"""Polars adapter: transforms and predicates as Polars expressions.

The expressions plug into the native Polars and deltalake calls::

    write_deltalake(dt, df.with_columns(partition_columns(dt)), mode="append")
    pl.scan_delta(dt).filter(predicate(dt, {"post_pt_root_id": roots}))

``scan_partitionwise`` builds the filtered scans of each step of a partition-wise
join.

This adapter calls Polars directly, not narwhals. Narwhals cannot express the parts
that differ by engine (see "no narwhals" in plans/decisions.md).
"""

import operator
from collections.abc import Iterator, Mapping, Sequence
from functools import reduce

import polars as pl
import pyarrow as pa
from deltalake import DeltaTable

from .partition_spec import PartitionSpec
from .table import as_partition_spec, iter_partitionwise, partition_values
from .transforms import IcebergBucket, to_arrow_int64


def partition_expr(expr: pl.Expr | str, transform: IcebergBucket) -> pl.Expr:
    """``transform`` applied to an integer expression, with the transform's type.

    Use it to add a partition column to a frame that does not have it, for example
    before a partition-wise join with a table that has a partition spec.
    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(
        lambda series: pl.Series(transform.apply(series.to_arrow())),
        return_dtype=pl.Series(pa.array([], transform.PARTITION_TYPE)).dtype,
        is_elementwise=True,
    )


def partition_columns(table: DeltaTable | PartitionSpec) -> list[pl.Expr]:
    """One expression per partition field of ``table``, named as its partition column.

    ``table`` is a ``DeltaTable`` or its partition spec. Pass the result to
    ``with_columns`` on a frame that has the source columns, then write the frame
    with ``write_deltalake``.
    """
    spec = as_partition_spec(table)
    spec.require_known()
    return [partition_expr(f.source, f.transform).alias(f.name) for f in spec.fields]


def _in(column: str, values: Sequence[int]) -> pl.Expr:
    return pl.col(column).is_in(list(values))


def _source_values(values: Sequence[int]) -> list[int]:
    # uint64 values >= 2**63 become their int64 view, as Delta stores them.
    return [v for v in to_arrow_int64(values).to_pylist() if v is not None]


def predicate(table: DeltaTable | PartitionSpec, where: Mapping[str, Sequence[int]]) -> pl.Expr:
    """A filter that keeps the rows whose source columns have the values in ``where``.

    ``where`` maps source columns to the values to keep. The filter tests the
    partition columns, so that ``pl.scan_delta`` opens only the matching files, and
    the source columns, so that only the matching rows stay.
    """
    predicates = [_in(name, v) for name, v in partition_values(table, where).items()]
    predicates += [_in(column, _source_values(values)) for column, values in where.items()]
    return reduce(operator.and_, predicates) if predicates else pl.lit(True)


def scan_partitionwise(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[pl.LazyFrame]]]:
    """For each partition of a partition-wise join, one scan per table.

    See ``cavebear.table.iter_partitionwise``. Join the scans of one partition, then
    move on to the next. Memory then holds one partition at a time.
    """
    for value, filters in iter_partitionwise(tables, on, n):
        yield value, [
            pl.scan_delta(dt).filter(*(_in(name, v) for name, v in nested.items()))
            for dt, nested in zip(tables, filters)
        ]
