# CAVE Delta bucketing — decisions

Upcoming Delta-based additions to the CAVE stack will lay out tables using deterministic transforms, such as hash buckets on IDs and spatial partitions. Delta has no notion of partition transforms or bucketing, so there is currently no standard way for a table to record how it was laid out. A reader has to know separately which columns were hashed, with what function, into how many buckets. This document collects decisions toward a small standard that fills that gap. It reuses existing schemes where they fit, such as Iceberg's bucket transform and Neuroglancer's sharding spec, so that generic engines like Polars and DuckDB can prune efficiently. It also describes a small Python utility package that serves as the reference implementation of the standard and as a helper for creating, writing and reading these tables consistently.

- Agreed: We will define an Iceberg-style bucket transform standard for Delta tables in the CAVE stack.
  - Agreed: For IDs, the spec should support both the existing Iceberg bucket standard and Neuroglancer's sharding spec as options.
    - Agreed: `num_buckets` is a power of two, so buckets nest across tables: a table with fewer buckets can be matched to one with more using a modulo.
  - TODO: For spatial data, Morton and Hilbert curves are options. Any spatial transform has more fields to pin down than ID bucketing.
    - Agreed: The spatial transform should nest across tables, so cells in a table with a coarser grid map cleanly onto cells in a table with a finer one. It should also nest within the ChunkedGraph.
    - TODO: Look into which standards and ideas for spatial partitioning we can borrow directly from Neuroglancer.
- Agreed: Layout metadata is table-level: every file in a table shares one layout, because changing a layout in Delta means rewriting all the data anyway.
  - Agreed: Layout metadata has exactly one source of truth. It lives either in the table metadata or in a sidecar, never in both.
  - TODO: Define a standard for where to put this metadata, for both ID and spatial transforms, because Delta has no equivalent place.
- Agreed: A small `cavebear` library will provide a core with test tables and reference implementations, plus optional backends such as `polars` or `duckdb`.
  - Agreed: It will include optional tooling for the full table lifecycle: creating a new table with a given spec, writing to it correctly (for example, computing the partition columns with an expression before writing), and reading from it efficiently (for example, adding the partition column to a join or pushing down a filter).
- Agreed: Bucket values are stored in the data as real partition columns. They compress to almost nothing, and they let generic engines like Polars and DuckDB prune files, which is a requirement.