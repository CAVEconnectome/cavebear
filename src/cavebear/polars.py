"""Polars adapter: transforms and predicates as Polars expressions.

The expressions plug into the native Polars and deltalake calls::

    write_deltalake(dt, df.with_columns(partition_columns(dt)), mode="append")
    pl.scan_delta(dt).filter(predicate(dt, {"post_pt_root_id": roots}))

For a partition-wise join, ``partitionwise_join_predicates`` gives one filter per table
for each step, for the native scan::

    for value, preds in partitionwise_join_predicates(tables, on):
        scans = [pl.scan_delta(dt).filter(pred) for dt, pred in zip(tables, preds)]

``scan_delta_partitionwise_join`` is a shortcut for the scans.
"""

import operator
from collections.abc import Iterator, Mapping, Sequence
from functools import reduce

import polars as pl
import pyarrow as pa
from deltalake import DeltaTable

from .partition_spec import PartitionSpec
from .table import as_partition_spec, partition_values, partitionwise_join_values
from .transforms import IcebergBucket, to_arrow_int64


def partition_expr(expr: pl.Expr | str, transform: IcebergBucket) -> pl.Expr:
    """``transform`` applied to an integer expression, with the transform's type.

    Use it to add a partition column to a frame that does not have it, for example
    before a partition-wise join with a table that has a partition spec.

    Parameters
    ----------
    expr
        An integer expression, or the name of an integer column.
    transform
        The transform to apply.

    Returns
    -------
    expr :
        The partition values, with the ``PARTITION_TYPE`` of the transform.
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

    Pass the result to ``with_columns`` on a frame that has the source columns, then
    write the frame with ``write_deltalake``.

    Parameters
    ----------
    table
        A Delta table or its partition spec.

    Returns
    -------
    exprs :
        One expression per field, in spec order.

    Raises
    ------
    ValueError
        If a field has a transform that cavebear does not know.
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

    The filter tests the partition columns, so that ``pl.scan_delta`` opens only the
    matching files, and the source columns, so that only the matching rows stay.

    Parameters
    ----------
    table
        A Delta table or its partition spec.
    where
        Maps source columns to the values to keep.

    Returns
    -------
    predicate :
        The filter, for ``LazyFrame.filter``.
    """
    predicates = [_in(name, v) for name, v in partition_values(table, where).items()]
    predicates += [_in(column, _source_values(values)) for column, values in where.items()]
    return reduce(operator.and_, predicates) if predicates else pl.lit(True)


def partitionwise_join_predicates(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[pl.Expr]]]:
    """For each step of a partition-wise join, one filter per table.

    See ``cavebear.table.partitionwise_join_values``, which gives the same steps as
    lists of partition values. Put each filter on ``pl.scan_delta`` of its table, and
    join the scans of one step.

    Parameters
    ----------
    tables
        The tables to join.
    on
        The source column of the join in each table, or one name for all of them.
    n
        The partition count of the steps, as in
        ``cavebear.table.partitionwise_join_values``.

    Yields
    ------
    value :
        The partition value of the step.
    predicates :
        One filter per table. Each keeps the partitions of its table that are
        nested in the partition of the step, for example
        ``pl.col("post_pt_root_id_partition").is_in([3, 19, 35, 51])``.
    """
    for value, filters in partitionwise_join_values(tables, on, n):
        yield value, [
            reduce(operator.and_, (_in(name, v) for name, v in nested.items()))
            for nested in filters
        ]


def scan_delta_partitionwise_join(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[pl.LazyFrame]]]:
    """For each step of a partition-wise join, one scan per table.

    Join the scans of one step, then move on to the next. Memory then holds one
    partition at a time.

    Parameters
    ----------
    tables
        The tables to join.
    on
        The source column of the join in each table, or one name for all of them.
    n
        The partition count of the steps, as in
        ``cavebear.table.partitionwise_join_values``.

    Yields
    ------
    value :
        The partition value of the step.
    scans :
        One ``pl.scan_delta`` per table, with the filters of
        ``partitionwise_join_predicates``.
    """
    for value, preds in partitionwise_join_predicates(tables, on, n):
        yield value, [pl.scan_delta(dt).filter(pred) for dt, pred in zip(tables, preds)]
