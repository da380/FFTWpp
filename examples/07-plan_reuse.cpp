#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

// This example shows how to reuse one plan across many buffers, and what the
// rule is for doing so safely.
//
// Planning with Measure or Patient is expensive; executing is not. A program
// that transforms many arrays of the same shape should plan once. FFTW allows
// that through its new-array execute functions, but only for buffers that
// match the ones the plan was made for in two respects: the same number of
// elements, and the same alignment class. FFTW does not check either, and
// getting it wrong is undefined behaviour rather than a wrong answer you can
// see. FFTWpp can check for you.

int main() {
  using namespace FFTWpp;

  using Real = double;
  using Complex = std::complex<Real>;

  constexpr int n = 256;

  //--------------------------------------------------------------------//
  //                        Plan once, run often                        //
  //--------------------------------------------------------------------//
  {
    // The buffers used for planning. With Measure their contents are
    // destroyed while planning, so nothing useful is put in them yet.
    auto planningIn = vector<Complex>(n);
    auto planningOut = vector<Complex>(n);
    auto plan = Ranges::Plan(Ranges::View(planningIn),
                             Ranges::View(planningOut), Measure, Forward);

    // Storage allocated through FFTWpp::vector always shares an alignment
    // class, so any of these may be substituted for the planning buffers.
    auto batch = std::vector<vector<Complex>>{};
    for (int i = 0; i < 4; ++i) {
      auto data = vector<Complex>(n);
      RandomiseValues(data);
      batch.push_back(std::move(data));
    }

    auto results = std::vector<vector<Complex>>(4, vector<Complex>(n));
    for (int i = 0; i < 4; ++i) {
      // ExecuteChecked verifies the rule before running; Execute skips the
      // check, and asserts it in a debug build. Prefer the checked form until
      // the surrounding code is settled.
      plan.ExecuteChecked(batch[i], results[i]);
    }

    // Confirm one of them against a plan made for those buffers directly.
    auto reference = vector<Complex>(n);
    auto direct = Ranges::Plan(Ranges::View(batch[0]), Ranges::View(reference),
                               Estimate, Forward);
    direct.Execute();
    if (!CheckValues(reference, results[0], Complex{1})) {
      std::cerr << "07-plan_reuse: a reused plan gave a different answer\n";
      return EXIT_FAILURE;
    }
    std::cout << "07-plan_reuse: one plan transformed " << batch.size()
              << " buffers, matching a plan built for them\n";
  }

  //--------------------------------------------------------------------//
  //                     Asking before executing                        //
  //--------------------------------------------------------------------//
  {
    auto in = vector<Complex>(n);
    auto out = vector<Complex>(n);
    auto plan =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Estimate, Forward);

    // The alignment class is FFTW's own notion, exposed here so that a
    // consumer can reason about it rather than guess.
    std::cout << "07-plan_reuse: the plan was built for alignment classes "
              << plan.InputAlignment() << " and " << plan.OutputAlignment()
              << '\n';

    // A buffer of the wrong length is rejected.
    auto tooShort = vector<Complex>(n / 2);
    if (plan.CanExecuteOn(tooShort, out)) {
      std::cerr << "07-plan_reuse: a short buffer was accepted\n";
      return EXIT_FAILURE;
    }
    try {
      plan.ExecuteChecked(tooShort, out);
      std::cerr << "07-plan_reuse: ExecuteChecked ran on a short buffer\n";
      return EXIT_FAILURE;
    } catch (const std::invalid_argument& error) {
      std::cout << "07-plan_reuse: short buffer rejected -- " << error.what()
                << '\n';
    }

    // A sub-span that starts partway into a buffer usually has a different
    // alignment class, and is rejected for that reason rather than its size.
    //
    // Real buffers are used here rather than complex ones, because FFTW's
    // alignment granularity may be no finer than one complex number, in which
    // case shifting a complex buffer by an element would not change its class
    // at all.
    auto realIn = vector<Real>(n);
    auto realOut = vector<Real>(n);
    auto realPlan = Ranges::Plan(Ranges::View(realIn), Ranges::View(realOut),
                                 Estimate, DHT);

    auto oversized = vector<Real>(n + 1);
    auto shifted = std::span<Real>(oversized.data() + 1, n);

    if (SameAlignment(oversized.data(), oversized.data() + 1)) {
      std::cout << "07-plan_reuse: this FFTW build does not distinguish the "
                   "alignment of a shifted pointer\n";
    } else {
      if (realPlan.CanExecuteOn(shifted, realOut)) {
        std::cerr << "07-plan_reuse: a differently aligned buffer was "
                     "accepted\n";
        return EXIT_FAILURE;
      }
      std::cout << "07-plan_reuse: a shifted sub-span was rejected on "
                   "alignment, not size\n";

      // Planning with Unaligned tells FFTW to assume nothing about
      // alignment, so the same buffer becomes usable. The cost is the SIMD
      // kernels the aligned plan could have used, so this is worth it only
      // when the storage really is outside your control.
      auto relaxed = Ranges::Plan(Ranges::View(realIn), Ranges::View(realOut),
                                  Estimate | Unaligned, DHT);
      if (!relaxed.CanExecuteOn(shifted, realOut)) {
        std::cerr << "07-plan_reuse: an unaligned plan still refused the "
                     "shifted buffer\n";
        return EXIT_FAILURE;
      }
      relaxed.ExecuteChecked(shifted, realOut);
      std::cout << "07-plan_reuse: an Unaligned plan accepted the same "
                   "buffer\n";
    }
  }

  //--------------------------------------------------------------------//
  //                      Copying and moving plans                      //
  //--------------------------------------------------------------------//
  {
    // A Plan owns its FFTW handle and destroys it. Copying builds a second,
    // independent plan from the same description; moving hands the handle
    // over and leaves the source empty. Neither leaks, and neither destroys
    // a handle twice.
    auto in = vector<Complex>(n);
    auto out = vector<Complex>(n);
    auto original =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Estimate, Forward);

    auto duplicate = original;
    auto moved = std::move(original);
    if (duplicate.Pointer() == moved.Pointer() || !original.IsNull()) {
      std::cerr << "07-plan_reuse: plan ownership did not behave as "
                   "documented\n";
      return EXIT_FAILURE;
    }
    std::cout << "07-plan_reuse: " << LivePlanCount()
              << " plans are alive at this point\n";
  }

  std::cout << "07-plan_reuse: every check passed\n";

  // Note that FFTWpp::CleanUp() is deliberately not called. FFTW's persistent
  // state is reachable for the life of the process, so leaving it is not a
  // leak, and discarding it would throw away the wisdom this run accumulated.
  return EXIT_SUCCESS;
}
