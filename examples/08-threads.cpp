#include <FFTWpp/Ranges>
#include <atomic>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

// This example shows the two quite different things "threads" can mean here,
// and which one you probably want.
//
// The first is planning and executing from several threads at once, each on
// its own data. FFTW's planner is not re-entrant, so this would be a race --
// but FFTWpp holds a process-wide lock across every planner call it makes, so
// there is nothing to do. That is the common case, and it needs no options.
//
// The second is splitting a single large transform across threads. That is
// FFTW's own threading, it is opt-in, and it is the right answer only when
// there is one transform big enough to be worth dividing.

int main() {
  using namespace FFTWpp;

  using Real = double;
  using Complex = std::complex<Real>;

  //--------------------------------------------------------------------//
  //          Many threads, each with its own transform                 //
  //--------------------------------------------------------------------//
  {
    constexpr int threadCount = 4;
    constexpr int transformsPerThread = 8;
    constexpr int n = 512;

    std::atomic<int> correct{0};
    auto workers = std::vector<std::jthread>();

    for (int t = 0; t < threadCount; ++t) {
      workers.emplace_back([&, t]() {
        // Each thread plans and executes independently. No lock is taken
        // here, and none is needed: FFTWpp takes one inside every planner
        // call. Execution is deliberately left unlocked, since that is where
        // the work is and FFTW's execute routines are thread-safe.
        auto in = vector<Complex>(n);
        auto out = vector<Complex>(n);
        auto copy = vector<Complex>(n);

        auto forward = Ranges::Plan(Ranges::View(in), Ranges::View(out),
                                    Estimate, Forward);
        auto backward = Ranges::Plan(Ranges::View(out), Ranges::View(copy),
                                     Estimate, Backward);

        for (int i = 0; i < transformsPerThread; ++i) {
          RandomiseValues(in, static_cast<std::uint64_t>(t * 100 + i));
          auto original = in;
          forward.Execute();
          backward.Execute();
          if (CheckValues(original, copy, backward.Normalisation())) {
            ++correct;
          }
        }
      });
    }
    workers.clear();  // joins every thread

    const auto expected = threadCount * transformsPerThread;
    if (correct.load() != expected) {
      std::cerr << "08-threads: only " << correct.load() << " of " << expected
                << " concurrent round trips matched\n";
      return EXIT_FAILURE;
    }
    std::cout << "08-threads: " << expected << " round trips across "
              << threadCount << " threads, with no lock in this file\n";
  }

  //--------------------------------------------------------------------//
  //             Mixing FFTWpp with the FFTW C API                      //
  //--------------------------------------------------------------------//
  {
    // If some of your code calls FFTW directly, serialise it against the same
    // lock. PlannerLock is the RAII form; PlannerMutex() is the mutex itself,
    // for cases the RAII form does not fit. The lock is not recursive, so do
    // not hold it across a call back into FFTWpp.
    auto in = vector<Complex>(64);
    auto out = vector<Complex>(64);

    fftw_plan raw = nullptr;
    {
      auto lock = PlannerLock{};
      raw = fftw_plan_dft_1d(64, reinterpret_cast<fftw_complex*>(in.data()),
                             reinterpret_cast<fftw_complex*>(out.data()),
                             FFTW_FORWARD, FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
      std::cerr << "08-threads: the raw plan could not be created\n";
      return EXIT_FAILURE;
    }
    fftw_execute(raw);
    {
      auto lock = PlannerLock{};
      fftw_destroy_plan(raw);
    }
    std::cout << "08-threads: a raw FFTW plan was serialised against the same "
                 "mutex\n";
  }

  //--------------------------------------------------------------------//
  //          One large transform, split across threads                 //
  //--------------------------------------------------------------------//
  if constexpr (ThreadsEnabled) {
    // This branch is compiled away entirely when the build did not link
    // FFTW's threading libraries, so the calls below cost nothing -- and
    // cause no link error -- in a build that could not satisfy them. Turn it
    // on with -DFFTWPP_USE_FFTW_THREADS=ON.
    constexpr int n = 1 << 16;

    // The session initialises FFTW's threading, sets how many threads a plan
    // may use, and cleans up when it goes out of scope. Create it before any
    // plan and let it outlive every plan: the cleanup requires that all plans
    // have been destroyed, and FFTWpp checks that rather than let a live plan
    // be quietly invalidated.
    auto session = ThreadSession(2);

    auto in = vector<Complex>(n);
    auto out = vector<Complex>(n);
    auto copy = vector<Complex>(n);

    auto forward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure, Forward);
    auto backward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure, Backward);

    RandomiseValues(in);
    auto original = in;
    forward.Execute();
    backward.Execute();

    if (!CheckValues(original, copy, backward.Normalisation())) {
      std::cerr << "08-threads: the threaded round trip did not match\n";
      return EXIT_FAILURE;
    }
    std::cout << "08-threads: a transform of length " << n
              << " ran across two FFTW threads\n";
  } else {
    std::cout << "08-threads: FFTW's own threading is not enabled in this "
                 "build; configure with -DFFTWPP_USE_FFTW_THREADS=ON to "
                 "include it\n";
  }

  std::cout << "08-threads: every check passed\n";

  return EXIT_SUCCESS;
}
