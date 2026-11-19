"""DuckDB adapter: scans with partition pruning, and the bucket transform as a SQL function.

The scans use ``delta_scan(uri, version => n)`` at the version of the given
``DeltaTable``, so DuckDB reads the same snapshot as the partition spec. The filters
are literal ``IN`` lists, so that DuckDB pushes them into the scan and skips files.
"""

from collections.abc import Iterator, Mapping, Sequence

import duckdb
import pyarrow as pa
from deltalake import DeltaTable

from .table import iter_partitionwise, partition_values
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


def _scan_sql(dt: DeltaTable, predicates: Sequence[str]) -> str:
    uri = dt.table_uri.replace("'", "''")
    sql = f"SELECT * FROM delta_scan('{uri}', version => {dt.version()})"
    return sql + (" WHERE " + " AND ".join(predicates) if predicates else "")


def scan(
    con: duckdb.DuckDBPyConnection,
    dt: DeltaTable,
    where: Mapping[str, Sequence[int]] | None = None,
) -> duckdb.DuckDBPyRelation:
    """A relation on the snapshot ``dt``, filtered to ``where``.

    ``where`` maps source columns to the values to keep. The relation filters on the
    partition values of those values, so that DuckDB opens only their files, and on the
    source columns, so that only the matching rows stay.
    """
    predicates = []
    if where:
        predicates += [_in(name, v) for name, v in partition_values(dt, where).items()]
        predicates += [
            _in(column, [v for v in to_arrow_int64(values).to_pylist() if v is not None])
            for column, values in where.items()
        ]
    return con.sql(_scan_sql(dt, predicates))


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
