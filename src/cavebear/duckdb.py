"""DuckDB adapter: transforms and predicates as DuckDB SQL.

The SQL plugs into the native DuckDB and deltalake calls::

    register(con)
    query = f"SELECT *, {partition_columns(dt)} FROM synapses"
    write_deltalake(dt, con.sql(query), mode="append")
    con.sql(f"SELECT * FROM delta_scan('{uri}') WHERE {predicate(dt, where)}")

``scan`` and ``scan_partitionwise`` are shortcuts for the second form.

The scans use ``delta_scan(uri, version => n)`` at the version of the given
``DeltaTable``, so DuckDB reads the same snapshot as the partition spec. The filters
are literal ``IN`` lists, so that DuckDB pushes them into the scan and skips files.
"""

from collections.abc import Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
from deltalake import DeltaTable

from .partition_spec import PartitionSpec
from .table import as_partition_spec, iter_partitionwise, partition_values
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
    result after ``SELECT *,`` in a query on rows that have the source columns, then
    write the relation with ``write_deltalake``.
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


def scan(
    con: duckdb.DuckDBPyConnection,
    dt: DeltaTable,
    where: Mapping[str, Sequence[int]] | None = None,
) -> duckdb.DuckDBPyRelation:
    """A ``delta_scan`` relation on the snapshot ``dt``, filtered with
    ``predicate(dt, where)``."""
    return con.sql(_scan_sql(dt, [predicate(dt, where)] if where else []))


def scan_partitionwise(
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[DeltaTable],
    on: str | Sequence[str],
    n: int | None = None,
) -> Iterator[tuple[int, list[duckdb.DuckDBPyRelation]]]:
    """For each partition of a partition-wise join, one relation per table.

    See ``cavebear.table.iter_partitionwise``. Join the relations of one partition,
    then move on to the next.
    """
    for value, filters in iter_partitionwise(tables, on, n):
        yield value, [
            con.sql(_scan_sql(dt, [_in(name, v) for name, v in nested.items()]))
            for dt, nested in zip(tables, filters)
        ]
