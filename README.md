# FFTWpp: A Modern C++ Wrapper for FFTW3

[![CI](https://github.com/da380/FFTWpp/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/FFTWpp/actions/workflows/ci.yml)

**FFTWpp** is a header-only C++20 library that provides a modern, type-safe
wrapper around [FFTW3](http://www.fftw.org/). It uses concepts and ranges to
give an expressive and safe API, and takes on the parts of FFTW's contract
that are easy to get wrong: precision dispatch, plan lifetime, planner
thread-safety and new-array execution.

### Key features

* **Modern C++ idioms.** Concepts for compile-time validation, ranges and
  views for data handling, on a C++20 baseline that works with both
  libstdc++ and libc++.
* **RAII plan management.** `FFTWpp::Ranges::Plan` owns its `fftw_plan` and
  destroys it at the end of its scope.
* **Type safety.** `Flag`, `Direction` and `RealKind` replace raw integers
  and enums.
* **Automatic precision selection.** The `fftwf_`, `fftw_` and `fftwl_`
  families are chosen from your data types; you never name one.
* **The advanced interface, wrapped.** `Layout` describes strided and batched
  transforms as a `(count, stride, dist)` descriptor, so a tensor can often be
  transformed in place rather than repacked.
* **Thread-safe by construction.** FFTWpp serialises FFTW's non-re-entrant
  planner itself. See [Thread safety](#thread-safety).

---

## Installing and consuming

### As an installed package

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build
cmake --install build
```

Then, from a consuming project:

```cmake
find_package(FFTWpp REQUIRED)
target_link_libraries(YourTarget PRIVATE FFTWpp::FFTWpp)
```

The installed package brings its FFTW find module with it, so
`find_package(FFTWpp)` resolves the `FFTW::*` targets without any help from
the consumer.

### As a subproject

`add_subdirectory` and `FetchContent` both work, and both give the same
`FFTWpp::FFTWpp` target:

```cmake
include(FetchContent)
FetchContent_Declare(
  FFTWpp
  GIT_REPOSITORY https://github.com/da380/FFTWpp.git
  GIT_TAG main
  FIND_PACKAGE_ARGS NAMES FFTWpp)
FetchContent_MakeAvailable(FFTWpp)
target_link_libraries(YourTarget PRIVATE FFTWpp::FFTWpp)
```

`FIND_PACKAGE_ARGS` is worth the one line: `FetchContent_MakeAvailable` then
tries `find_package` first, so an installed copy is used when one exists and
only a genuinely absent dependency is downloaded.

### Build options

| Option | Default | Meaning |
| --- | --- | --- |
| `FFTWPP_BUILD_TESTS` | on at top level | Build the GoogleTest suite |
| `FFTWPP_BUILD_EXAMPLES` | on at top level | Build the examples |
| `FFTWPP_BUILD_DOCS` | `OFF` | Build the Doxygen documentation |
| `FFTWPP_INSTALL` | on at top level | Generate install and export rules |
| `FFTWPP_WARNINGS_AS_ERRORS` | `OFF` | Treat warnings in FFTWpp's own targets as errors |
| `FFTWPP_USE_FFTW_THREADS` | `OFF` | Link FFTW's threading libraries |
| `FFTWPP_THREAD_BACKEND` | `Threads` | `Threads` (pthreads) or `OpenMP` |

### Requirements

A C++20 compiler and FFTW3 built for all three precisions: `libfftw3`,
`libfftw3f` and `libfftw3l`. On Debian and Ubuntu that is `libfftw3-dev`; on
macOS, `brew install fftw`.

GCC 11, Clang 16 and Apple Clang 15 or newer should all work; CI covers GCC
13, Clang 18 and the Apple Clang on the current macOS runner. The library
deliberately avoids the C++23 ranges algorithms — `fold_left_first` and
`zip_transform` — because libc++ does not ship them until LLVM 22 and 23,
which would have excluded macOS entirely.

---

## Core concepts

### Layout and View

FFTWpp separates the *shape* of your data from the data itself.

* `FFTWpp::Ranges::Layout` describes the shape: rank, dimensions, how many
  transforms, the embedded (physical) dimensions, the stride between elements
  and the distance between transforms. These are exactly the parameters of
  FFTW's advanced interface.
* `FFTWpp::Ranges::View` pairs a `Layout` with a range of data. It is what you
  hand to a `Plan`.

Both validate their arguments when they are constructed and throw
`std::invalid_argument` if they cannot describe a transform. The checks are
unconditional rather than assertions, so a release build reports the problem
instead of passing it to FFTW.

### Plan

`FFTWpp::Ranges::Plan` is an RAII wrapper around an `fftw_plan`. Construct it
with input and output views; it creates the plan, `Execute()` runs it, and the
destructor calls `fftw_destroy_plan`. Plans are copyable — a copy is a new
plan built from the same description — and movable, which transfers the handle
and leaves the source null.

### Options

* `FFTWpp::Direction` — `Forward`, `Backward`.
* `FFTWpp::Flag` — `Estimate`, `Measure`, `Patient`, `Exhaustive`,
  `WisdomOnly`, `DestroyInput`, `PreserveInput`, `Unaligned`, combined with
  `|` or accumulated with `|=`.
* `FFTWpp::RealKind` — the real-to-real kinds, with `Inverse()` and
  `LogicalDimension(n)` for normalisation.

---

## Quick start

```cpp
#include <FFTWpp/Ranges>

#include <algorithm>
#include <complex>
#include <iostream>

int main() {
  constexpr int n = 16;
  using Complex = std::complex<double>;

  // FFTWpp::vector is std::vector with FFTW's aligned allocator.
  auto in = FFTWpp::vector<Complex>(n);
  auto transformed = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                      FFTWpp::Ranges::View(transformed),
                                      FFTWpp::Measure, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(transformed),
                                       FFTWpp::Ranges::View(out),
                                       FFTWpp::Measure, FFTWpp::Backward);

  // Measure overwrites its arrays while planning, so fill them in afterwards.
  FFTWpp::RandomiseValues(in);
  const auto original = in;

  forward.Execute();
  backward.Execute();

  // A backward transform is unnormalised; Normalisation() is the factor.
  if (FFTWpp::CheckValues(original, out, backward.Normalisation())) {
    std::cout << "the inverse transform matches the original data\n";
  }

}
```

`Measure` and the more thorough flags time candidate algorithms on your actual
arrays, and overwrite them in the process, which is why the data is filled in
only after the plans exist. `Estimate` does not, and can be planned in any
order.

---

## Advanced usage

### Real transforms

The transform type follows from the value types of the views.

```cpp
// Real to complex: the complex side has n / 2 + 1 elements.
auto realData = FFTWpp::vector<double>(n);
auto complexData = FFTWpp::vector<std::complex<double>>(n / 2 + 1);
auto planR2C = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(realData),
                                    FFTWpp::Ranges::View(complexData),
                                    FFTWpp::Measure);

// Real to real: give the kind, or one kind per dimension.
auto planDCT = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(realIn),
                                    FFTWpp::Ranges::View(realOut),
                                    FFTWpp::Measure, FFTWpp::REDFT10);
```

`FFTWpp::DataSize<InType, OutType>(dimensions...)` returns the pair of storage
sizes a given transform needs, and is a constant expression.

### Batched and strided transforms

A `Layout` built with the full advanced-interface signature describes many
transforms at once:

```cpp
// howMany transforms of length n, each in its own contiguous block.
auto layout = FFTWpp::Ranges::Layout(/*rank=*/1, std::vector{n}, howMany,
                                     /*embed=*/std::vector{n},
                                     /*stride=*/1, /*dist=*/n);

// The interleaved alternative: element i of transform j at j + howMany * i.
auto interleaved = FFTWpp::Ranges::Layout(1, std::vector{n}, howMany,
                                          std::vector{n}, howMany, 1);

auto plan = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, layout),
                                 FFTWpp::Ranges::View(out, layout),
                                 FFTWpp::Measure, FFTWpp::Forward);
```

Describing the layout costs a descriptor rather than a repack, which is the
reason to wrap the advanced interface at all.

### New-array execution and alignment

A plan may be executed on buffers other than the ones it was created for, but
only if those buffers hold the same number of elements *and* share FFTW's
alignment class. FFTW does not check this, and neither does the plain
overload:

```cpp
plan.Execute(otherIn, otherOut);          // fast, unchecked
plan.ExecuteChecked(otherIn, otherOut);   // throws if it would be invalid
if (plan.CanExecuteOn(otherIn, otherOut)) { /* ask in advance */ }
```

`FFTWpp::AlignmentOf(pointer)` and `FFTWpp::SameAlignment(first, second)`
expose FFTW's own `fftw*_alignment_of`; `plan.InputAlignment()` and
`plan.OutputAlignment()` report the classes the plan was created with. In a
debug build `Execute(in, out)` asserts the same condition, so the unchecked
path is still diagnosable while developing.

Buffers allocated through `FFTWpp::vector` or `FFTWpp::Allocator` always share
an alignment class, so the common case needs no thought. Sub-spans of a larger
buffer are where this matters.

---

## Thread safety

FFTW's planner is not re-entrant: apart from the execution routines, no two
FFTW calls may run concurrently. **FFTWpp handles this for you.** A
process-wide mutex is taken around every planner call the library makes — plan
creation, plan destruction, wisdom manipulation and `CleanUp()`. A consumer
that plans from several threads needs no lock of its own, and a consumer that
carries one may delete it.

Execution is deliberately left unlocked, since `fftw_execute` is thread-safe
and is where the work happens.

If you mix FFTWpp with direct calls into the FFTW C API, take the same lock:

```cpp
{
  auto lock = FFTWpp::PlannerLock{};
  auto raw = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
}
```

`FFTWpp::PlannerMutex()` is the underlying `std::mutex` for cases the RAII
form does not fit. The lock is not recursive, so do not hold it across a call
into FFTWpp itself.

### FFTW's own threading

FFTW can also split a *single* transform across threads, which is a different
thing from the above and is **opt-in**. It is off by default because a
consumer that already parallelises over many independent transforms — the
common case in this library's intended use — wants each plan single-threaded,
and nesting the two is worse than either.

Turn it on with `-DFFTWPP_USE_FFTW_THREADS=ON`, which links FFTW's threading
libraries and defines `FFTWPP_ENABLE_THREADS`. Choose the backend with
`-DFFTWPP_THREAD_BACKEND=Threads` (pthreads, the default) or `OpenMP`.

```cpp
// Initialises FFTW threading, allows four threads per plan, and cleans up
// when it goes out of scope. Create it before any plan and let it outlive
// every plan.
auto threads = FFTWpp::ThreadSession(4);

auto plan = FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Measure,
                                 FFTWpp::Forward);
plan.Execute();
```

`InitialiseThreads()`, `PlanWithNumberOfThreads(n)` and `CleanUpThreads()` are
available directly if the RAII form does not suit. `FFTWpp::ThreadsEnabled` is
a compile-time constant reporting whether this build linked the threading
libraries, so a portable consumer can branch on it rather than on the macro.

---

## Wisdom

Wisdom lets FFTW reuse what it learned about one transform when planning the
same shape again, which is what makes `Measure` and `Patient` affordable at
start-up.

```cpp
// At start-up: load what a previous run learned.
try {
  FFTWpp::ImportWisdom<double>("fftw-wisdom-double.dat");
} catch (const std::runtime_error&) {
  // No wisdom yet. Planning still works; it is just slower this once.
}

// Optionally pre-plan the shapes you know you will need.
auto layout = FFTWpp::Ranges::Layout(128, 128);
FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Patient);

// ... run the application ...

// At exit: keep what this run learned.
FFTWpp::ExportWisdom<double>("fftw-wisdom-double.dat");
```

A few things are worth knowing:

* **Wisdom is per precision.** FFTW keeps a separate store, in a separate file
  format, for `float`, `double` and `long double`. `ExportWisdom<Real>` and
  `ImportWisdom<Real>` name the precision; the un-templated `ExportWisdom` and
  `ImportWisdom` are the double-precision ones. An application using more than
  one precision needs one file per precision.
* **A missing shape is not an error.** If a plan's shape is absent from the
  wisdom, FFTW simply plans it from scratch, and the result joins the wisdom
  for next time. The exception is `WisdomOnly`, which refuses to plan anything
  not already in the wisdom and so throws — useful for asserting that
  pre-generation covered everything, and a trap otherwise.
* **`GenerateWisdom` with `Estimate` does nothing**, since `Estimate` performs
  no measurements to remember.

`examples/Example5.cpp` walks through the whole cycle and is run as part of
the test suite.

`ExportWisdomToString<Real>()` and `ImportWisdomFromString<Real>(text)` serve
the same purpose without a file. `ImportSystemWisdom<Real>()` reads the
machine-wide wisdom an administrator may have generated, and reports whether
it found any rather than throwing. `ForgetWisdom()` clears every precision;
`ForgetWisdom<Real>()` clears one.

### On `CleanUp()`

**Most programs should not call `FFTWpp::CleanUp()`.** It exists, but it is not
the way a program using FFTWpp is supposed to finish. Only `Example5` calls
it, and only to demonstrate the third case below.

FFTW keeps persistent state — accumulated wisdom and the list of algorithms
available in this configuration — in its own globals, reachable for the life of
the process. Leaving it there is not a leak, and no leak checker reports one.
Calling `CleanUp()` is worth it in three situations:

1. Under a leak checker configured to report *still reachable* blocks, when a
   silent report is wanted. This is why the test suite calls it.
2. In a plugin or extension module that may be unloaded from a long-lived host
   process, where the state really would be orphaned.
3. To reset FFTW deliberately, for instance to force re-measurement.

Two costs come with it. It **discards accumulated wisdom**, so any
`ExportWisdom` must happen first. And it leaves **every live plan undefined** —
including plans owned by unrelated code in the same process, which is why a
library should be reluctant to call it on its users' behalf.

FFTWpp counts the plans it owns and `CleanUp()` throws `std::logic_error`
rather than let that happen silently; `LivePlanCount()` reports the same
number. The check cannot see raw handles obtained from `Core.h` and owned by
the caller, so it is a necessary condition rather than a proof.
`CleanUpThreads()` carries the same caveats and the same check.

---

## Testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure

./scripts/test_sanitized.sh address   # ASan + UBSan
./scripts/test_sanitized.sh thread    # TSan; checks the planner mutex
```

See [docs/testing.md](docs/testing.md) for the test inventory.

---

## Library structure

| Header | Contents |
| --- | --- |
| `Core.h` | Precision-aware wrappers over `fftw3.h`, the aligned allocator, the planner mutex, alignment queries and the threading wrappers |
| `Options.h` | `Direction`, `Flag`, `RealKind` |
| `Views.h` | `Ranges::Layout` and `Ranges::View` |
| `Plan.h` | `Ranges::Plan` |
| `Wisdom.h` | Wisdom import, export and generation |
| `Utility.h` | `DataSize`, `RandomiseValues`, `CheckValues` |

Include `<FFTWpp/Ranges>` for the whole library, or `<FFTWpp/Core>` for the
low-level wrappers and the allocator alone.
