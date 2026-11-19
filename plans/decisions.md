# CAVE Delta partitioning — decisions

Upcoming Delta-based additions to the CAVE stack will lay out tables using deterministic transforms, such as hash buckets on IDs and spatial partitions. Delta has no notion of partition transforms or bucketing, so there is currently no standard way for a table to record how it was partitioned. A reader has to know separately which columns were hashed, with what function, into how many buckets. This document collects decisions toward a small standard that fills that gap. It reuses existing schemes where they fit, such as Iceberg's bucket transform and Neuroglancer's sharding spec, so that generic engines like Polars and DuckDB can prune efficiently. It also describes a small Python utility package that serves as the reference implementation of the standard and as a helper for creating, writing and reading these tables consistently.

- Agreed: We will define an Iceberg-style partition transform standard for Delta tables in the CAVE stack.
  - Agreed: For IDs, the standard should support both the existing Iceberg bucket transform and Neuroglancer's sharding spec as options.
    - Agreed: The partition count is a power of two, so partitions nest across tables: a table with fewer partitions can be matched to one with more using a modulo.
    - Agreed: The Neuroglancer option puts each row in the partition of the shard file that holds its ID in an existing precomputed dataset, so a table lines up with the meshes or skeletons of the same IDs.
  - TODO: For spatial data, Morton and Hilbert curves are options. Any spatial transform has more fields to pin down than ID partitioning.
    - Agreed: The spatial transform should nest across tables, so cells in a table with a coarser grid map cleanly onto cells in a table with a finer one. It should also nest within the ChunkedGraph.
    - TODO: Look into which standards and ideas for spatial partitioning we can borrow directly from Neuroglancer.
- Agreed: The partition spec is table-level: every file in a table shares one partition spec, because changing it in Delta means rewriting all the data anyway.
  - Agreed: The partition spec has exactly one source of truth. It lives either in the table metadata or in a sidecar, never in both.
  - Agreed: The partition spec lives in the Delta table `configuration` under the key `cave.partition_spec`, not in a sidecar file. It commits with the data, so a reader always gets the partition spec of the version it reads, and engines already cache it with the snapshot.
    - Agreed: deltalake rejects custom keys only by default. `DeltaTable.create(..., raise_if_key_not_exists=False)` accepts them, and all writes and maintenance keep the key (`tests/test_delta_configuration.py`).
    - TODO: Revisit Delta `domainMetadata` once Python `deltalake` can write it.
  - Agreed: A transform is an object with a `@type` member, so the `sharding` object of a Neuroglancer precomputed `info` file can be copied into a partition spec without a change.
  - Agreed: A reader warns on a transform with an unknown `@type` and does not use it for pruning, because the reads still give correct rows and only open more files.
- Agreed: A small `cavebear` library will provide a core with test tables and reference implementations, plus optional backends such as `polars` or `duckdb`.
  - Agreed: It will include optional tooling for the full table lifecycle: creating a new table with a given partition spec, writing to it correctly (for example, computing the partition columns with an expression before writing), and reading from it efficiently (for example, adding the partition column to a join or pushing down a filter).
  - Agreed: The Arrow stream (`__arrow_c_stream__`) is the shared write path, because pyarrow, Polars and DuckDB all export it, so one `write` adds the partition columns for every engine.
  - Agreed: The engine adapters call Polars and DuckDB directly, not narwhals, because narwhals has no DuckDB `map_batches`, no integer XOR or bit shift, and no Delta scan.
  - Agreed: The transforms run as a numpy kernel on Arrow arrays, with `pyiceberg_core` as the test oracle, because numpy is already a dependency and the oracle checks the kernel against the Iceberg reference.
- Agreed: Partition values are stored in the data as real partition columns. They compress to almost nothing, and they let generic engines like Polars and DuckDB prune files, which is a requirement.
