"""DuckDB adapter: transforms and predicates as DuckDB SQL.

The SQL plugs into the native DuckDB and deltalake calls, in a query or in the
relation API::

    register(con)
    write_deltalake(dt, rel.project(f"*, {partition_columns(dt)}"), mode="append")
    con.sql(f"FROM delta_scan('{uri}')").filter(predicate(dt, where))

The filters are literal ``IN`` lists, so that DuckDB pushes them into the scan and
skips files.

For a partition-wise join, ``partitionwise_join_predicates`` gives one filter per table
for each step, for the native scan::

    for value, preds in partitionwise_join_predicates(tables, on):
        scans = [
            con.sql(f"FROM delta_scan('{dt.table_uri}')").filter(pred)
            for dt, pred in zip(tables, preds)
        ]

``scan_delta_partitionwise_join`` is a shortcut for the scans. Its scans use
``delta_scan(uri, version => n)`` at the version of each ``DeltaTable``, so DuckDB
reads the same snapshot as the partition spec.
"""

from collections.abc import Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
from deltalake import DeltaTable

from .partition_spec import PartitionSpec
from .table import as_partition_spec, partition_values, partitionwise_join_values
from .transforms import IcebergBucket, to_arrow_int64

#: The name of the SQL function that ``register`` adds.
FUNCTION = "cave_iceberg_bucket"


def register(con: duckdb.DuckDBPyConnection) -> None:
    """Add ``cave_iceberg_bucket(value BIGINT, n_buckets INTEGER) -> INTEGER`` to ``con``.

    A ``UBIGINT`` value at or above 2**63 does not fit ``BIGINT``. Cast such a column
    to its ``BIGINT`` view first.
    """

    def function(values: pa.ChunkedArray, n_buckets: pa.ChunkedArray) -> pa.ChunkedArray:
        counts = set(n_buckets.to_pylist()) - {None}
        if len(counts) > 1:
            raise ValueError(f"{FUNCTION} needs one n_buckets per query, got {sorted(counts)}")
        n = counts.pop() if counts else 1
        return IcebergBucket(n).apply(values)

    con.create_function(
        FUNCTION,
        function,
        [duckdb.sqltype("BIGINT"), duckdb.sqltype("INTEGER")],
        duckdb.sqltype("INTEGER"),
        type="arrow",
    )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _in(column: str, values: Sequence[int]) -> str:
    literals = ", ".join(str(v) for v in values)
    return f"{_quote_identifier(column)} IN ({literals})" if values else "FALSE"


def partition_columns(table: DeltaTable | PartitionSpec) -> str:
    """A SQL select list with one expression per partition field of ``table``.

    ``table`` is a ``DeltaTable`` or its partition spec. Each expression calls the
    function that ``register`` adds, and is named as its partition column. Put the
    result after ``*,`` in a select list or in ``rel.project``, on rows that have the
    source columns, then write the relation with ``write_deltalake``.
    """
    spec = as_partition_spec(table)
    spec.require_known()
    return ", ".join(
        f"{FUNCTION}({_quote_identifier(f.source)}, {f.transform.n_buckets}) "
        f"AS {_quote_identifier(f.name)}"
        for f in spec.fields
    )


def predicate(table: DeltaTable | PartitionSpec, where: Mapping[str, Sequence[int]]) -> str:
    """A SQL condition that keeps the rows whose source columns have the values in
    ``where``.

    ``where`` maps source columns to the values to keep. The condition tests the
    partition columns, so that ``delta_scan`` opens only the matching files, and the
    source columns, so that only the matching rows stay. The values are literal
    ``IN`` lists, so that DuckDB pushes them into the scan.
    """
    predicates = [_in(name, v) for name, v in partition_values(table, where).items()]
    predicates += [
        _in(column, [v for v in to_arrow_int64(values).to_pylist() if v is not None])
        for column, values in where.items()
    ]
    return " AND ".join(predicates) if predicates else "TRUE"


def _scan_sql(dt: DeltaTable, predicates: Sequence[str]) -> str:
    uri = dt.table_uri.replace("'", "''")
    sql = f"SELECT * FROM delta_scan('{uri}', version => {dt.version()})"
    return sql + (" WHERE " + " AND ".join(predicates) if predicates else "")


def partitionwise_join_predicates(
    tables: Sequence[DeltaTable], on: str | Sequence[str], n: int | None = None
) -> Iterator[tuple[int, list[str]]]:
    """For each step of a partition-wise join, one SQL condition per table.

    See ``cavebear.table.partitionwise_join_values``, which gives the same steps as lists of
    partition values. Each condition keeps the partitions of its table that are nested
    in the partition of the step, for example
    ``"post_pt_root_id_bucket" IN (3, 19, 35, 51)``. Put each condition on a
    ``delta_scan`` of its table, and join the scans of one step.
    """
    for value, filters in partitionwise_join_values(tables, on, n):
        yield value, [" AND ".join(_in(name, v) for name, v in nested.items()) for nested in filters]


def scan_delta_partitionwise_join(
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[DeltaTable],
    on: str | Sequence[str],
    n: int | None = None,
) -> Iterator[tuple[int, list[duckdb.DuckDBPyRelation]]]:
    """For each step of a partition-wise join, one relation per table.

    The relations are ``delta_scan`` at the version of each table, with the
    conditions of ``partitionwise_join_predicates``. Join the relations of one step, then
    move on to the next.
    """
    for value, preds in partitionwise_join_predicates(tables, on, n):
        yield value, [con.sql(_scan_sql(dt, [pred])) for dt, pred in zip(tables, preds)]
