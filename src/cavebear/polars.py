"""Polars adapter: scans with partition pruning, and transforms as expressions.

This adapter calls Polars directly, not narwhals. Narwhals cannot express the parts
that differ by engine (see "no narwhals" in plans/decisions.md).
"""

from collections.abc import Iterator, Mapping, Sequence

import polars as pl
import pyarrow as pa
from deltalake import DeltaTable

from .table import iter_partitionwise, partition_values
from .transforms import IcebergBucket, to_arrow_int64


def _source_filter(column: str, values: Sequence[int]) -> pl.Expr:
    # uint64 values >= 2**63 become their int64 view, as Delta stores them.
    return pl.col(column).is_in(to_arrow_int64(values).to_pylist())


def scan(dt: DeltaTable, where: Mapping[str, Sequence[int]] | None = None) -> pl.LazyFrame:
    """A lazy scan of the snapshot ``dt``, filtered to ``where``.

    ``where`` maps source columns to the values to keep. The scan filters on the
    partition values of those values, so that Polars opens only their files, and on
    the source columns, so that only the matching rows stay.
    """
    lf = pl.scan_delta(dt)
    if not where:
        return lf
    predicates = [pl.col(name).is_in(values) for name, values in partition_values(dt, where).items()]
    predicates += [_source_filter(column, values) for column, values in where.items()]
    return lf.filter(*predicates)


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


def scan_partitionwise(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[pl.LazyFrame]]]:
    """For each partition of a partition-wise join, one scan per table.

    See ``cavebear.table.iter_partitionwise``. Join the scans of one partition, then
    move on to the next. Memory then holds one partition at a time.
    """
    for value, filters in iter_partitionwise(tables, on, n):
        yield value, [
            scan(dt).filter(*(pl.col(name).is_in(v) for name, v in nested.items()))
            for dt, nested in zip(tables, filters)
        ]
