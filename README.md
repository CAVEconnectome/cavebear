# cistern

CAVE 🤝 data lakehouse

## Why

The [Connectome Annotation Versioning Engine (CAVE)](https://github.com/CAVEconnectome) is increasingly going to use [data lakehouse](https://www.databricks.com/blog/what-is-data-lakehouse)-style tooling such as [Delta Lake](https://delta.io/) or [Apache Iceberg](https://iceberg.apache.org/).

Part of the advantage of these approaches is that they allow us to leverage client-side
query engines (such as [Polars](https://pola.rs/) or [DuckDB](https://duckdb.org/)) to handle much of the compute associated with readin and writing data. This means that we don't have to write these complicated tools from scratch
and can benefit from common tooling with other applications. However, there are some small
pieces of the CAVE stack that are still semi-unique or not well supported by these tools,
such as how hashing functions or spatial chunking are handled.

`cistern` therefore has two goals: 1) to document specs for these standards and
serve as a reference implementation thereof (a concrete example is a spec for how to hash
long IDs (like root IDs) for partitioning), and 2) to provide lightweight utility functions
and adapters that let query engines leverage any of these new standards without the user
having to think too hard about them.

The lakehouse ecosystem continues to evolve, so we could also see some of the custom tooling
we are building here being more generally supported in the future, so parts of this package may
cease to be necessary and can be offloaded to other tools.

## Roadmap

- [ ] Document the specs that are needed. To begin with, hash and spatial bucketing rules.
- [ ] Create reference implementations that adhere to those specs.
- [ ] Create utility functions for creating, reading, and comparing specs for things like nestedness.
- [ ] Create table expressions that compute the approproate transforms.
- [ ] Create an API that enables query engines to appropriately prune queries based on transformed data.

## Collaboration

This is a work in progress, we are actively exploring this data lakehouse space in the context of connectome data. Feedback, questions, or ideas are welcome! Also note that the tools here are preliminary and likely to change.
