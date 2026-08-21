@mainpage FFTWpp

@tableofcontents

**FFTWpp** is a header-only C++20 library that wraps
[FFTW3](http://www.fftw.org/). It uses concepts and ranges to give a
type-safe, expressive API, and takes on the parts of FFTW's contract that are
easy to get wrong: precision dispatch, plan lifetime, planner thread-safety
and new-array execution.

Include `<FFTWpp/Ranges>` for the whole library, or `<FFTWpp/Core>` for the
low-level wrappers and the aligned allocator alone.

@section overview Where to look

| If you want | Start at |
| --- | --- |
| The transform interface | FFTWpp::Ranges::Plan |
| To describe your data's shape | FFTWpp::Ranges::Layout, FFTWpp::Ranges::View |
| Flags, directions and r2r kinds | FFTWpp::Direction, FFTWpp::Flag, FFTWpp::RealKind |
| Aligned storage | FFTWpp::Allocator, FFTWpp::vector |
| Planner thread-safety | FFTWpp::PlannerLock, FFTWpp::PlannerMutex |
| FFTW's own threading | FFTWpp::ThreadSession, FFTWpp::ThreadsEnabled |
| Alignment and new-array execution | FFTWpp::AlignmentOf, FFTWpp::Ranges::Plan::ExecuteChecked |
| Wisdom | FFTWpp::ImportWisdom, FFTWpp::ExportWisdom, FFTWpp::GenerateWisdom |

@section quickstart Quick start

@code{.cpp}
#include <FFTWpp/Ranges>

#include <complex>
#include <iostream>

int main() {
  constexpr int n = 16;
  using Complex = std::complex<double>;

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

  if (FFTWpp::CheckValues(original, out, backward.Normalisation())) {
    std::cout << "the inverse transform matches the original data\n";
  }
}
@endcode

@section structure Design

@subsection layout_view Layout and View

FFTWpp::Ranges::Layout describes the *shape* of the data: rank, dimensions,
how many transforms, the embedded (physical) dimensions, the stride between
elements and the distance between transforms. These are exactly the parameters
of FFTW's advanced interface, so a batched or strided transform costs a
descriptor rather than a repack.

FFTWpp::Ranges::View pairs a layout with a range of data, and is what a plan
is constructed from. Both validate their arguments unconditionally and throw
`std::invalid_argument` rather than asserting, so a release build reports a
bad shape instead of handing it to FFTW.

@subsection plan Plan

FFTWpp::Ranges::Plan owns an `fftw_plan` and destroys it in its destructor.
Copying builds an equivalent new plan; moving transfers the handle and leaves
the source null. FFTWpp::Ranges::Plan::Normalisation gives the factor an
unnormalised inverse transform needs.

@subsection threads Thread safety

FFTW's planner is not re-entrant. FFTWpp serialises it: a process-wide mutex
is held across every planner call the library makes, so consumers need no lock
of their own. Execution is left unlocked. Use FFTWpp::PlannerLock to serialise
your own direct calls into the FFTW C API against the same mutex.

FFTW's ability to split a *single* transform across threads is a separate,
opt-in feature; see FFTWpp::ThreadSession and the CMake option
`FFTWPP_USE_FFTW_THREADS`.

@subsection alignment Alignment

A plan may be executed on new buffers only if they match the planning buffers
in size and in FFTW's alignment class. FFTWpp::AlignmentOf exposes the class,
FFTWpp::Ranges::Plan::CanExecuteOn asks whether a substitution is valid, and
FFTWpp::Ranges::Plan::ExecuteChecked throws rather than invoking undefined
behaviour.

@subsection cleanup Finishing

Most programs should not call FFTWpp::CleanUp. FFTW's persistent state is
reachable for the life of the process, so leaving it is not a leak and no leak
checker reports one; calling it discards accumulated wisdom and leaves every
live plan undefined. FFTWpp::LivePlanCount reports how many plans would be
affected, and CleanUp refuses rather than doing it silently. See the
FFTWpp::CleanUp documentation for the three situations that do want it.

@section more Further reading

The repository's `README.md` covers installation, CMake integration and
wisdom management at greater length, and `docs/testing.md` lists what the test
suite covers.
