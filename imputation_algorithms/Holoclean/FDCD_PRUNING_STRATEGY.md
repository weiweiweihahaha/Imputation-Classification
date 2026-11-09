# FDCD Pre-Enumeration Pruning Strategy

This document describes the pruning strategy used by
`code/scripts/generate_holoclean_constraints.py`
before Holoclean's bundled FDCD discovery enters the DC enumeration stage.

## Background

The original pipeline sent all feature columns directly into FDCD and only applied
`max_predicates` and `max_constraints` after Java had already finished discovery.
That meant the following failure mode was still possible:

- FDCD builds a large predicate space.
- FDCD builds the evidence set.
- FDCD runs INCS enumeration and minimality checking.
- Java exhausts heap memory before Python post-filters the discovered constraints.

For datasets such as `heart_disease`, this caused `java.lang.OutOfMemoryError: Java heap space`
inside the INCS minimality check.

## Goal

Move pruning earlier so that search-space reduction happens before FDCD starts
enumeration, rather than after all raw denial constraints have already been generated.

## Where Pruning Happens

Pruning is applied in `_prepare_discovery_frame(...)`, before the temporary FDCD input CSV
is written to `Denial_Constraint_Discovery/generated_inputs/`.

This means the reduced feature set is what FDCD actually reads, so the smaller predicate
space affects:

- predicate-space construction
- evidence-set construction
- DC enumeration
- minimality checking

## Pruning Steps

The current strategy has three stages.

### 1. Drop empty or constant columns

Each feature column is profiled first.

A column is dropped immediately if its number of distinct non-empty values is `<= 1`.

This removes:

- all-empty columns
- effectively constant columns
- columns that carry no useful variation for DC discovery

### 2. Score remaining columns by usefulness

Each surviving column receives a lightweight score based on:

- non-null ratio
- number of distinct non-empty values
- uniqueness ratio
- a small preference for numeric columns

The intent is not to estimate the exact best subset, but to keep columns that are more
informative and less likely to be pure noise.

### 3. Enforce a pre-enumeration predicate budget

FDCD creates predicates per column roughly as follows:

- categorical column: 2 predicates
  - `EQ`
  - `IQ`
- numeric column: 6 predicates
  - `EQ`
  - `IQ`
  - `LT`
  - `LTE`
  - `GT`
  - `GTE`

Using this cost model, the script greedily keeps the highest-ranked columns while the
estimated predicate budget is still available.

If a column would push the selected set over budget, it is dropped before discovery starts.

## Parameters

The script now supports two pre-enumeration controls.

### `--pre_enum_max_predicates`

Estimated predicate-space budget applied before FDCD runs.

- numeric columns cost `6`
- categorical columns cost `2`
- default: `48`
- set to `0` to disable

This is the main safety valve against INCS memory blow-ups.

### `--pre_enum_max_columns`

Hard cap on the number of feature columns retained before generating the FDCD input CSV.

- default: disabled
- set a positive integer to enforce a strict column cap

This is useful when you want an explicit structural bound in addition to the predicate budget.

## Metadata Output

For each dataset, the generated `*_constraints_meta.json` now records:

- original feature count
- feature count after pruning
- estimated predicate count before pruning
- estimated predicate count after pruning
- selected columns
- dropped columns
- drop reasons

This makes it easy to inspect exactly why a dataset was reduced before enumeration.

## Example: `heart_disease`

With the original 13 feature columns, FDCD built 78 predicates and could run out of heap
memory during INCS enumeration.

With pre-enumeration pruning enabled:

- input columns were reduced from 13 to 8
- estimated predicates were reduced from 78 to 48
- FDCD enumeration completed successfully

## Relationship to Existing Post-Filters

The existing options below still remain useful:

- `--max_predicates`
- `--max_constraints`

But these are still post-processing filters on the discovered raw DC output.

They do not protect the Java enumeration stage from exploding in memory.

In short:

- pre-enumeration pruning reduces search cost
- post-enumeration filtering reduces final output size

Both can be used together, but they solve different problems.

## Recommended Usage

For unstable or memory-heavy datasets:

```bash
conda run -n holoclean_py310 python code/scripts/generate_holoclean_constraints.py \
  --datasets heart_disease \
  --skip_build \
  --algorithm INCS \
  --pre_enum_max_predicates 48 \
  --max_predicates 4 \
  --max_constraints 5000 \
  --java_xmx 32g
```

If a dataset is still expensive, try one or more of:

- lower `--pre_enum_max_predicates`
- add `--pre_enum_max_columns`
- switch `--algorithm` from `INCS` to `HEI` or `EI`

## Current Scope

This strategy is implemented on the Python side by pruning columns before writing the FDCD
input CSV.

It does not yet modify FDCD's Java internals to prune predicate groups after predicate-space
construction. If deeper control is needed later, that can be added as a second layer.
