<p align="left"><img src="docs/assets/cavebear-lockup.svg" alt="cavebear" height="180"></p>

CAVE 🤝 data lakehouse table formats

## Why

The [Connectome Annotation Versioning Engine (CAVE)](https://github.com/CAVEconnectome) is increasingly going to use [data lakehouse](https://www.databricks.com/blog/what-is-data-lakehouse)-style tooling, for instance, CAVE currently uses [Delta Lake](https://delta.io/) and may support [Apache Iceberg](https://iceberg.apache.org/). Both store large tables in parquet files written to object storage, but in a manner that makes them queryable and updatable by a variety of tools.

Part of the advantage of these approaches is that they allow us to leverage client-side
query engines (such as [Polars](https://pola.rs/) or [DuckDB](https://duckdb.org/)) to handle much of the compute associated with reading and writing data. This means that we don't have to write these complicated tools from scratch
and can benefit from common tooling. However, there are some small
pieces of the CAVE stack that are still semi-unique or not widely supported by these tools,
such as how hashing functions or spatial chunking are handled.

`cavebear` therefore has two goals: 1) to document specs for these standards and
serve as a reference implementation thereof (a concrete example is a spec for how to hash
long IDs for partitioning), and 2) to provide lightweight utility functions
and adapters that let query engines leverage any of these new standards without the user
having to think too hard about them.

The lakehouse ecosystem continues to evolve, so we could also see some of the custom tooling
we are building here being more generally supported in the future, so parts of this package may cease to be necessary and can be offloaded to other tools.

## Roadmap

- [x] Document a spec for storing metadata about Iceberg-style ID -> partition hashing in the Delta Lake metadata.
- [x] Create implementations for reading from and writing to Delta Lake tables with that partition hashing scheme.
- [ ] Create tooling for partition-aware joins across tables, possibly with different partition specs.
- [ ] Create a spec for storing spatial partitions nested within the ChunkedGraph spatial partitions.

## Collaboration

This is a work in progress, and we are actively exploring this data lakehouse space in the context of connectome data. Feedback, questions, or ideas are welcome! Also note that the tools here are preliminary and likely to change.
