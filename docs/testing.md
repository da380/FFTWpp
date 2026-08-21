# Testing

The `Tests` executable uses GoogleTest, and CTest discovers each typed
precision case separately. Its custom `main` calls `FFTWpp::CleanUp()` only
after all tests, and therefore all test-local plans, have finished — one of
the few places that call is worth making, since the sanitizer workflow wants
FFTW's still-reachable state released so that anything left over is this
library's doing. The five examples are registered as tests too, since each
verifies its own behaviour and reports the result through its exit status.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Sanitizers

```bash
./scripts/test_sanitized.sh address   # AddressSanitizer + UndefinedBehaviorSanitizer
./scripts/test_sanitized.sh thread    # ThreadSanitizer
```

The two cannot be combined, so they are separate runs, and CI runs both.

The address run is what catches the leak and double-destruction regressions
the plan-ownership tests are written against. The thread run is what
demonstrates that the planner mutex does its job: with the lock removed from
`Core.h`, `PlannerSerialisation.ConcurrentPlanCreationCompletes` reports a
data race inside FFTW reached through `Plan::MakePlan`; with it in place there
are no reports at all.

FFTW itself is linked uninstrumented, so ThreadSanitizer cannot see the
synchronisation inside it. In practice it stays quiet and no suppression file
is needed; set `TSAN_SUPPRESSIONS` to one if a particular FFTW build turns out
to need it. A `called_from_lib` entry must match exactly one loaded library or
ThreadSanitizer refuses to start, so name the full soname — `libfftw3.so.3`,
not `libfftw3`, which also matches `libfftw3f.so.3`.

On recent Linux kernels GCC's ThreadSanitizer refuses to start because the
kernel's mmap randomisation is wider than it expects; the script runs `ctest`
under `setarch -R` to avoid it.

## Test inventory

### `Tests.cpp` — plan ownership and wisdom

- `PlanOwnership.CopyConstructionAndAssignmentOwnDistinctPlans` — copying
  creates an independent FFTW plan, and copy assignment replaces an
  already-owned destination plan.
- `PlanOwnership.MoveConstructionAndAssignmentTransferOwnership` — move
  construction and assignment transfer the exact FFTW handle, leave the source
  null, and replace an already-owned destination.
- `PlanOwnership.SelfAndRepeatedAssignmentRemainValid` — self-copy, self-move,
  and repeated assignment of both kinds.
- `PlanOwnership.FailedCopyReplacementPreservesDestination` — removing the
  wisdom a `WisdomOnly` plan needs makes the copy fail, and the destination
  keeps its existing handle.
- `WisdomGeneration.*` — generated wisdom lets the complex, real-complex and
  real-to-real plans be recreated with `WisdomOnly`, in both directions.
- `WisdomIO.DoublePrecisionExportAndImportReportSuccessCorrectly` — the
  double-precision file API and its return handling.

### `TestTransforms.cpp` — round-trip correctness

Every case runs for `float`, `double` and `long double`, and checks the
reported `Normalisation()` as well as the recovered values.

- `TwoDimensionalComplexRoundTrip`, `ThreeDimensionalRealToComplexRoundTrip`,
  `TwoDimensionalRealToRealRoundTripPerDimensionKinds` — multi-dimensional
  transforms, including one r2r kind per dimension.
- `ASingleRealToRealKindAppliesToEveryDimension` — the documented rule that a
  short kind list is extended with its last entry.
- `BatchedContiguousComplexRoundTrip`, `BatchedInterleavedComplexRoundTrip`,
  `BatchedRealToComplexRoundTrip` — the advanced-interface layouts, in both
  the block and the interleaved arrangement. These are the shapes the
  `(count, stride, dist)` descriptor exists for.
- `InPlaceComplexRoundTrip` — input and output views over the same storage.

`ThreeDimensionalRealToComplexRoundTrip` also pins the normalisation fix: it
requires the forward and backward plans of one logical transform to report the
same factor, which an R2C plan did not before it used the real-space
dimensions.

### `TestSafety.cpp` — the diagnosable-failure surface

- `PlannerSerialisation.*` — many threads creating plans concurrently, and
  planning concurrent with execution. These pass trivially without the
  planner mutex; their value is under ThreadSanitizer.
- `LayoutValidation.*` — rank, dimension count, non-positive dimensions,
  embedded dimensions smaller than their dimension, zero transforms and zero
  stride are each rejected with `std::invalid_argument`.
- `ViewValidation`, `PlanValidation.*` — data that does not fill its layout,
  mismatched dimensions, a halfcomplex side of the wrong length, disagreeing
  transform counts, and too many r2r kinds.
- `Alignment.*` — `AlignmentOf` is stable and reflexive, agrees between real
  and complex views of one address, and distinguishes a shifted pointer.
- `NewArrayExecution.*` — `CanExecuteOn` and `ExecuteChecked` accept matching
  buffers and reject buffers of the wrong size or alignment class, and the
  new-array result matches the same transform run through the planning
  buffers. A plan created with `Unaligned` assumes nothing about alignment, so
  only its size rule is enforced.
- `Allocator.*` — allocation failure throws `std::bad_alloc` rather than
  returning null, the storage is FFTW-aligned, and instances compare equal.
- `LivePlanCount.*`, `CleanUp.RefusesToRunWhileAPlanIsAlive` — the count
  follows construction, destruction, copy, move and assignment, a failed plan
  is not counted, and `CleanUp` throws rather than leaving a live plan
  undefined. None of these calls `CleanUp` successfully: doing so mid-suite
  would discard the wisdom other tests rely on.
- `FftwThreads.*` — with `FFTWPP_USE_FFTW_THREADS=ON`, a `ThreadSession` plans
  and executes a large transform and a non-positive thread count is rejected.
  Otherwise a single test asserts that `ThreadsEnabled` is false.

### `TestGuru.cpp` — the guru interface

- `GuruLayout.*` — `RowMajorStrides`, that `TransformAlong` along an interior
  axis produces a two-dimensional batch, that `HalfcomplexShape` halves the
  last transformed axis, and that the real-to-complex and complex-to-real
  builders are mirror images.
- Validation: malformed dimensions, malformed axis lists, ranges too small for
  their layout, too many r2r kinds, and the overlap check — both that it
  catches two dimensions addressing the same element and that every layout
  `TransformAlong` produces passes it.
- `Guru.InteriorAxisAgreesWithTheTransformDoneByHand` is the load-bearing one.
  A transform along the middle axis of a three-dimensional array, done in a
  single guru plan, must agree element for element with the same transform
  driven by hand as one batched plan per slab — which is as far as the
  advanced interface reaches. A round trip cannot catch a wrong stride, since
  a consistently wrong forward and backward pair still round-trips.
- `GuruPlan.TransformsTheCallersBufferRatherThanACopyOfIt` guards the reason
  `GuruPlan` is constrained to views rather than ranges. When a container
  could be deduced it was passed by value, the plan was built on the copy's
  storage, and every transform wrote where the caller could not see it — which
  every round trip passed regardless.
- Round trips for complex, real-to-complex and real-to-real along an interior
  axis, at all three precisions, plus a transposing layout whose two sides have
  different stride orders.
- Ownership, checked new-array execution, and the predicate that chooses
  between FFTW's 32- and 64-bit guru entry points.

### `TestOptions.cpp` — options, utilities and wisdom routes

- `Options.*` — direction values, flag combination with `|` and `|=`, that
  every `RealKind` inverts back to itself, and the logical dimensions used for
  normalisation.
- `Utility.*` — `DataSize` as a constant expression, seeded `RandomiseValues`
  reproducibility, and `CheckValues` across differing range types and with an
  explicit tolerance.
- `Wisdom.*` — the string round trip, that each precision has its own store
  and that clearing one leaves the others, single-precision files, that a
  failed import names the precision and the file, and that generating with
  `Estimate` is a no-op.

### `OdrInclude.cpp`

Includes the public aggregate header in a second translation unit, so the
test link fails if a header-defined API function stops being inline.

### The examples

The examples are numbered in the order they are meant to be read, and each
returns a non-zero exit status if one of its own checks fails.

| | |
| --- | --- |
| `01-getting_started` | The same 1D transform through the FFTW C API, `Core.h` and the range interface |
| `02-transform_types` | Complex, real-to-complex and real-to-real |
| `03-many_dimensions` | Two, three and four dimensions, and per-dimension r2r kinds |
| `04-batched_transforms` | The advanced interface: contiguous and interleaved batches |
| `05-wisdom` | Loading at start-up, pre-generating shapes, `WisdomOnly` against an unknown shape, saving at exit |
| `06-guru_layouts` | Interior axes, several axes at once, halfcomplex, a hand-written descriptor, and the overlap check |
| `07-plan_reuse` | New-array execution, alignment classes, `Unaligned`, and plan ownership |
| `08-threads` | Concurrent planning, `PlannerLock` around the C API, and FFTW's own threading |

`08-threads` reports that FFTW-internal threading is unavailable rather than
skipping silently when the build did not link it; the branch is discarded by
`if constexpr`, so it costs nothing and causes no link error.

## Packaging

`tests/package` is a standalone project that consumes an installed FFTWpp
through `find_package`. It is not part of the main build; CI installs the
library to a prefix and then configures, builds and runs it against that
prefix, so a regression in the install or export rules is caught.
