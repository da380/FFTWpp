// Tests for the parts of the interface that exist to turn undefined
// behaviour into something diagnosable: planner serialisation, argument
// validation, alignment queries and allocation failure.
#include <gtest/gtest.h>

#include <FFTWpp/Ranges>
#include <atomic>
#include <complex>
#include <cstdint>
#include <new>
#include <span>
#include <thread>
#include <vector>

namespace {

using Complex = std::complex<double>;

//---------------------------------------------------------------------//
//                      Planner serialisation                          //
//---------------------------------------------------------------------//

// FFTW's planner is not re-entrant. These tests do the thing that used to be
// a race, and rely on the sanitizer workflow to say whether it still is.

TEST(PlannerSerialisation, ConcurrentPlanCreationCompletes) {
  constexpr int threadCount = 8;
  constexpr int plansPerThread = 16;
  std::atomic<int> created{0};

  auto workers = std::vector<std::jthread>();
  for (int t = 0; t < threadCount; ++t) {
    workers.emplace_back([&, t]() {
      for (int i = 0; i < plansPerThread; ++i) {
        const auto n = 8 + ((t * plansPerThread + i) % 24);
        auto in = FFTWpp::vector<Complex>(n);
        auto out = FFTWpp::vector<Complex>(n);
        auto plan = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                         FFTWpp::Ranges::View(out),
                                         FFTWpp::Estimate, FFTWpp::Forward);
        EXPECT_FALSE(plan.IsNull());
        ++created;
      }
    });
  }
  workers.clear();  // join

  EXPECT_EQ(created.load(), threadCount * plansPerThread);
}

TEST(PlannerSerialisation, ConcurrentPlanningAndExecutionAgree) {
  // Execution is deliberately left unlocked, so this also checks that
  // executing while another thread plans neither deadlocks nor corrupts.
  constexpr int n = 32;
  constexpr int threadCount = 4;

  auto workers = std::vector<std::jthread>();
  std::atomic<int> correct{0};
  for (int t = 0; t < threadCount; ++t) {
    workers.emplace_back([&]() {
      auto in = FFTWpp::vector<Complex>(n);
      auto out = FFTWpp::vector<Complex>(n);
      auto back = FFTWpp::vector<Complex>(n);
      auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                          FFTWpp::Ranges::View(out),
                                          FFTWpp::Estimate, FFTWpp::Forward);
      auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out),
                                           FFTWpp::Ranges::View(back),
                                           FFTWpp::Estimate, FFTWpp::Backward);
      for (int i = 0; i < 8; ++i) {
        FFTWpp::RandomiseValues(in);
        const auto original = in;
        forward.Execute();
        backward.Execute();
        if (FFTWpp::CheckValues(original, back, backward.Normalisation())) {
          ++correct;
        }
      }
    });
  }
  workers.clear();

  EXPECT_EQ(correct.load(), threadCount * 8);
}

TEST(PlannerSerialisation, TheMutexIsASingleProcessWideObject) {
  EXPECT_EQ(&FFTWpp::PlannerMutex(), &FFTWpp::PlannerMutex());
}

//---------------------------------------------------------------------//
//                        Layout validation                            //
//---------------------------------------------------------------------//

TEST(LayoutValidation, RejectsNonPositiveRank) {
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(0, std::vector{8}, 1, std::vector{8}, 1, 0),
      std::invalid_argument);
}

TEST(LayoutValidation, RejectsDimensionCountDisagreeingWithRank) {
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(2, std::vector{8}, 1, std::vector{8}, 1, 0),
      std::invalid_argument);
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(1, std::vector{8}, 1, std::vector{8, 8}, 1, 0),
      std::invalid_argument);
}

TEST(LayoutValidation, RejectsNonPositiveDimensions) {
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(1, std::vector{0}, 1, std::vector{8}, 1, 0),
      std::invalid_argument);
  EXPECT_THROW(FFTWpp::Ranges::Layout(8, 0), std::invalid_argument);
}

TEST(LayoutValidation, RejectsEmbeddedDimensionSmallerThanItsDimension) {
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(1, std::vector{8}, 1, std::vector{4}, 1, 0),
      std::invalid_argument);
}

TEST(LayoutValidation, RejectsNoTransformsAndZeroStride) {
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(1, std::vector{8}, 0, std::vector{8}, 1, 0),
      std::invalid_argument);
  EXPECT_THROW(
      FFTWpp::Ranges::Layout(1, std::vector{8}, 1, std::vector{8}, 0, 0),
      std::invalid_argument);
}

TEST(LayoutValidation, AcceptsABatchedAdvancedInterfaceLayout) {
  const auto layout =
      FFTWpp::Ranges::Layout(1, std::vector{16}, 10, std::vector{16}, 1, 16);
  EXPECT_EQ(layout.Rank(), 1);
  EXPECT_EQ(layout.HowMany(), 10);
  EXPECT_EQ(layout.size(), 160);
  EXPECT_EQ(layout.TransformSize(), 16);
}

//---------------------------------------------------------------------//
//                     View and Plan validation                        //
//---------------------------------------------------------------------//

TEST(ViewValidation, RejectsDataThatDoesNotFillItsLayout) {
  auto data = FFTWpp::vector<Complex>(8);
  const auto layout = FFTWpp::Ranges::Layout(16);
  EXPECT_THROW(FFTWpp::Ranges::View(data, layout), std::invalid_argument);
}

TEST(PlanValidation, RejectsMismatchedComplexDimensions) {
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(16);
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward),
      std::invalid_argument);
}

TEST(PlanValidation, RejectsAHalfcomplexSideOfTheWrongLength) {
  auto real = FFTWpp::vector<double>(8);
  auto wrong = FFTWpp::vector<Complex>(8);  // should be 8 / 2 + 1 == 5
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(real),
                           FFTWpp::Ranges::View(wrong), FFTWpp::Estimate),
      std::invalid_argument);
}

TEST(PlanValidation, RejectsDisagreeingTransformCounts) {
  auto in = FFTWpp::vector<Complex>(32);
  auto out = FFTWpp::vector<Complex>(32);
  const auto batched =
      FFTWpp::Ranges::Layout(1, std::vector{8}, 4, std::vector{8}, 1, 8);
  const auto single =
      FFTWpp::Ranges::Layout(1, std::vector{8}, 4, std::vector{8}, 1, 8);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, batched),
                                       FFTWpp::Ranges::View(out, single),
                                       FFTWpp::Estimate, FFTWpp::Forward));

  auto small = FFTWpp::vector<Complex>(16);
  const auto twoTransforms =
      FFTWpp::Ranges::Layout(1, std::vector{8}, 2, std::vector{8}, 1, 8);
  EXPECT_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, batched),
                                    FFTWpp::Ranges::View(small, twoTransforms),
                                    FFTWpp::Estimate, FFTWpp::Forward),
               std::invalid_argument);
}

TEST(PlanValidation, RejectsAnRealToRealPlanWithTooManyKinds) {
  auto in = FFTWpp::vector<double>(8);
  auto out = FFTWpp::vector<double>(8);
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate,
                           std::vector{FFTWpp::REDFT10, FFTWpp::REDFT10}),
      std::invalid_argument);
}

//---------------------------------------------------------------------//
//                             Alignment                               //
//---------------------------------------------------------------------//

TEST(Alignment, IsStableAndReflexive) {
  auto buffer = FFTWpp::vector<double>(16);
  EXPECT_EQ(FFTWpp::AlignmentOf(buffer.data()),
            FFTWpp::AlignmentOf(buffer.data()));
  EXPECT_TRUE(FFTWpp::SameAlignment(buffer.data(), buffer.data()));
}

TEST(Alignment, ComplexAndRealViewsOfTheSameAddressAgree) {
  auto buffer = FFTWpp::vector<Complex>(16);
  EXPECT_EQ(FFTWpp::AlignmentOf(buffer.data()),
            FFTWpp::AlignmentOf(reinterpret_cast<double*>(buffer.data())));
}

TEST(Alignment, AShiftedPointerHasADifferentClass) {
  auto buffer = FFTWpp::vector<double>(16);
  const auto base = FFTWpp::AlignmentOf(buffer.data());
  const auto shifted = FFTWpp::AlignmentOf(buffer.data() + 1);
  if (base == shifted) {
    GTEST_SKIP() << "this FFTW build does not distinguish alignment classes";
  }
  EXPECT_FALSE(FFTWpp::SameAlignment(buffer.data(), buffer.data() + 1));
}

TEST(NewArrayExecution, SucceedsOnBuffersMatchingThePlanningBuffers) {
  constexpr int n = 32;
  auto in = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);

  // Fresh fftw_malloc'd buffers share the planning buffers' alignment class.
  auto otherIn = FFTWpp::vector<Complex>(n);
  auto otherOut = FFTWpp::vector<Complex>(n);
  ASSERT_TRUE(plan.CanExecuteOn(otherIn, otherOut));
  EXPECT_NO_THROW(plan.ExecuteChecked(otherIn, otherOut));

  // The same transform, run through the planning buffers, must agree.
  FFTWpp::RandomiseValues(otherIn, std::uint64_t{4});
  std::ranges::copy(otherIn, in.begin());
  plan.ExecuteChecked(otherIn, otherOut);
  plan.Execute();
  EXPECT_TRUE(FFTWpp::CheckValues(out, otherOut, Complex{1}));
}

TEST(NewArrayExecution, RejectsBuffersOfTheWrongSize) {
  constexpr int n = 32;
  auto in = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);

  auto tooSmall = FFTWpp::vector<Complex>(n / 2);
  auto otherOut = FFTWpp::vector<Complex>(n);
  EXPECT_FALSE(plan.CanExecuteOn(tooSmall, otherOut));
  EXPECT_THROW(plan.ExecuteChecked(tooSmall, otherOut), std::invalid_argument);
}

TEST(NewArrayExecution, RejectsBuffersOfTheWrongAlignmentClass) {
  // Real buffers, because FFTW's alignment granularity may be no finer than
  // one complex number, in which case shifting a complex buffer by an element
  // would not change its class at all.
  constexpr int n = 32;
  auto in = FFTWpp::vector<double>(n);
  auto out = FFTWpp::vector<double>(n);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::DHT);

  auto oversized = FFTWpp::vector<double>(n + 1);
  if (FFTWpp::SameAlignment(oversized.data(), oversized.data() + 1)) {
    GTEST_SKIP() << "this FFTW build does not distinguish alignment classes";
  }
  auto shifted = std::span<double>(oversized.data() + 1, n);

  auto otherOut = FFTWpp::vector<double>(n);
  EXPECT_FALSE(plan.CanExecuteOn(shifted, otherOut));
  EXPECT_THROW(plan.ExecuteChecked(shifted, otherOut), std::invalid_argument);
}

TEST(NewArrayExecution, ThePlanReportsThePlanningBuffersAlignment) {
  constexpr int n = 32;
  auto in = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);
  EXPECT_EQ(plan.InputAlignment(), FFTWpp::AlignmentOf(in.data()));
  EXPECT_EQ(plan.OutputAlignment(), FFTWpp::AlignmentOf(out.data()));
  EXPECT_EQ(plan.PlannerFlag(), FFTWpp::Estimate);
}

//---------------------------------------------------------------------//
//                             Allocator                               //
//---------------------------------------------------------------------//

TEST(Allocator, ThrowsRatherThanReturningNullOnAnImpossibleRequest) {
  auto allocator = FFTWpp::Allocator<Complex>();
  Complex* p = nullptr;
  EXPECT_THROW(
      p = allocator.allocate(std::numeric_limits<std::size_t>::max() / 4),
      std::bad_alloc);
  EXPECT_EQ(p, nullptr);
}

TEST(Allocator, AllocatesFftwAlignedStorage) {
  auto allocator = FFTWpp::Allocator<double>();
  auto* p = allocator.allocate(64);
  ASSERT_NE(p, nullptr);
  auto reference = FFTWpp::vector<double>(64);
  EXPECT_TRUE(FFTWpp::SameAlignment(p, reference.data()));
  allocator.deallocate(p, 64);
}

TEST(Allocator, InstancesOfAnyValueTypeCompareEqual) {
  EXPECT_TRUE(FFTWpp::Allocator<double>() == FFTWpp::Allocator<Complex>());
  EXPECT_FALSE(FFTWpp::Allocator<double>() != FFTWpp::Allocator<Complex>());
}

//---------------------------------------------------------------------//
//                        Live plan accounting                         //
//---------------------------------------------------------------------//

// CleanUp leaves every live plan undefined, so the count of live plans is
// what makes calling it safe or not. None of these tests calls CleanUp
// successfully: doing so mid-suite would discard the wisdom other tests rely
// on, which is itself one of the reasons most programs should leave it alone.

TEST(LivePlanCount, TracksConstructionAndDestruction) {
  const auto before = FFTWpp::LivePlanCount();
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  {
    auto plan = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                     FFTWpp::Ranges::View(out),
                                     FFTWpp::Estimate, FFTWpp::Forward);
    EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);
  }
  EXPECT_EQ(FFTWpp::LivePlanCount(), before);
}

TEST(LivePlanCount, ACopyIsCountedSeparatelyAndAMoveIsNot) {
  const auto before = FFTWpp::LivePlanCount();
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);

  auto source =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Forward);
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);

  {
    auto copy = source;  // a second, independent FFTW plan
    EXPECT_EQ(FFTWpp::LivePlanCount(), before + 2);
  }
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);

  {
    auto moved = std::move(source);  // the same handle, moved
    EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);
  }
  EXPECT_EQ(FFTWpp::LivePlanCount(), before);
}

TEST(LivePlanCount, AssignmentReplacesRatherThanAccumulates) {
  const auto before = FFTWpp::LivePlanCount();
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto inView = FFTWpp::Ranges::View(in);
  auto outView = FFTWpp::Ranges::View(out);

  auto source =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Forward);
  auto destination =
      FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate, FFTWpp::Backward);
  ASSERT_EQ(FFTWpp::LivePlanCount(), before + 2);

  for (int i = 0; i < 8; ++i) destination = source;
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 2);

  for (int i = 0; i < 8; ++i) {
    auto temporary = FFTWpp::Ranges::Plan(inView, outView, FFTWpp::Estimate,
                                          FFTWpp::Backward);
    destination = std::move(temporary);
  }
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 2);
}

TEST(LivePlanCount, AFailedPlanIsNotCounted) {
  const auto before = FFTWpp::LivePlanCount();
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  FFTWpp::ForgetWisdom();
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::WisdomOnly, FFTWpp::Forward),
      std::runtime_error);
  EXPECT_EQ(FFTWpp::LivePlanCount(), before);
}

TEST(CleanUp, RefusesToRunWhileAPlanIsAlive) {
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);
  ASSERT_GT(FFTWpp::LivePlanCount(), 0);

  try {
    FFTWpp::CleanUp();
    FAIL() << "CleanUp should refuse to run while a plan is alive";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("still alive"), std::string::npos);
  }

  // The plan is untouched: refusing means refusing, not half cleaning up.
  EXPECT_FALSE(plan.IsNull());
  EXPECT_NO_THROW(plan.Execute());
}

#ifdef FFTWPP_ENABLE_THREADS

//---------------------------------------------------------------------//
//                     FFTW's own threading                            //
//---------------------------------------------------------------------//

TEST(FftwThreads, ASessionPlansAndExecutesALargeTransform) {
  ASSERT_TRUE(FFTWpp::ThreadsEnabled);
  constexpr int n = 4096;
  auto in = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);
  auto back = FFTWpp::vector<Complex>(n);

  {
    auto session = FFTWpp::ThreadSession(2);
    auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                        FFTWpp::Ranges::View(out),
                                        FFTWpp::Estimate, FFTWpp::Forward);
    auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out),
                                         FFTWpp::Ranges::View(back),
                                         FFTWpp::Estimate, FFTWpp::Backward);
    FFTWpp::RandomiseValues(in, std::uint64_t{11});
    const auto original = in;
    forward.Execute();
    backward.Execute();
    EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
  }
  // The session's destructor ran CleanUpThreads, so restore a usable planner
  // for the tests that follow.
  ASSERT_TRUE(FFTWpp::InitialiseThreads());
  FFTWpp::PlanWithNumberOfThreads(1);
}

TEST(FftwThreads, CleanUpThreadsRefusesToRunWhileAPlanIsAlive) {
  auto in = FFTWpp::vector<Complex>(8);
  auto out = FFTWpp::vector<Complex>(8);
  auto plan =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);
  EXPECT_THROW(FFTWpp::CleanUpThreads(), std::logic_error);
  EXPECT_FALSE(plan.IsNull());
}

TEST(FftwThreads, RejectsANonPositiveThreadCount) {
  EXPECT_THROW(FFTWpp::PlanWithNumberOfThreads(0), std::invalid_argument);
  EXPECT_THROW(FFTWpp::PlanWithNumberOfThreads(-1), std::invalid_argument);
}

#else

TEST(FftwThreads, AreReportedAsUnavailableInThisBuild) {
  EXPECT_FALSE(FFTWpp::ThreadsEnabled);
}

#endif

}  // namespace
