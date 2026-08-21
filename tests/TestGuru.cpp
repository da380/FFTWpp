// Tests for the guru interface. The point of it is layouts the advanced
// interface cannot express, so the load-bearing test is not a round trip but
// a cross-check: a transform along an interior axis, done in one guru plan,
// must agree element for element with the same transform driven by hand.
#include <gtest/gtest.h>

#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

using Complex = std::complex<double>;
using FFTWpp::Dim;

template <typename Real>
class Guru : public testing::Test {};

using Precisions = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(Guru, Precisions);

//---------------------------------------------------------------------//
//                       Building a layout                             //
//---------------------------------------------------------------------//

TEST(GuruLayout, RowMajorStridesAreTheProductsOfTrailingExtents) {
  EXPECT_EQ(FFTWpp::Ranges::RowMajorStrides({2, 3, 4}),
            (std::vector<std::ptrdiff_t>{12, 4, 1}));
  EXPECT_EQ(FFTWpp::Ranges::RowMajorStrides({7}),
            (std::vector<std::ptrdiff_t>{1}));
  EXPECT_THROW(static_cast<void>(FFTWpp::Ranges::RowMajorStrides({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(FFTWpp::Ranges::RowMajorStrides({2, 0})),
               std::invalid_argument);
}

TEST(GuruLayout, TransformAlongAnInteriorAxisBuildsATwoDimensionalBatch) {
  // The case the advanced interface cannot express: the repetitions start at
  // offsets i0 * n1 * n2 + i2, which is not an arithmetic progression.
  const auto layout = FFTWpp::Ranges::TransformAlong({2, 3, 4}, {1});

  ASSERT_EQ(layout.Rank(), 1);
  ASSERT_EQ(layout.BatchRank(), 2);
  EXPECT_EQ(layout.Transform()[0],
            (Dim{.n = 3, .inStride = 4, .outStride = 4}));
  EXPECT_EQ(layout.Batch()[0], (Dim{.n = 2, .inStride = 12, .outStride = 12}));
  EXPECT_EQ(layout.Batch()[1], (Dim{.n = 4, .inStride = 1, .outStride = 1}));
  EXPECT_EQ(layout.HowMany(), 8);
  EXPECT_EQ(layout.TransformSize(), 3);
  EXPECT_EQ(layout.Extent(FFTWpp::Ranges::Side::Input), 24);
}

TEST(GuruLayout, TransformAlongSeveralAxesUsesThemInMemoryOrder) {
  const auto layout = FFTWpp::Ranges::TransformAlong({2, 3, 4}, {2, 0});
  ASSERT_EQ(layout.Rank(), 2);
  EXPECT_EQ(layout.Transform()[0].n, 2);
  EXPECT_EQ(layout.Transform()[1].n, 4);
  ASSERT_EQ(layout.BatchRank(), 1);
  EXPECT_EQ(layout.Batch()[0].n, 3);
}

TEST(GuruLayout, HalfcomplexShapeHalvesTheLastTransformedAxis) {
  EXPECT_EQ(FFTWpp::Ranges::HalfcomplexShape({2, 3, 8}, {2}),
            (std::vector<std::ptrdiff_t>{2, 3, 5}));
  EXPECT_EQ(FFTWpp::Ranges::HalfcomplexShape({2, 8, 4}, {1}),
            (std::vector<std::ptrdiff_t>{2, 5, 4}));
  // With several axes it is the last of them that is halved.
  EXPECT_EQ(FFTWpp::Ranges::HalfcomplexShape({2, 8, 4}, {0, 1}),
            (std::vector<std::ptrdiff_t>{2, 5, 4}));
}

TEST(GuruLayout, RealToComplexKeepsRealExtentsAndHalfcomplexOutputStrides) {
  const auto layout = FFTWpp::Ranges::RealToComplexTransformAlong({4, 8}, {1});
  ASSERT_EQ(layout.Rank(), 1);
  // The extent is the real one, as FFTW's guru interface requires.
  EXPECT_EQ(layout.Transform()[0].n, 8);
  EXPECT_EQ(layout.Transform()[0].inStride, 1);
  EXPECT_EQ(layout.Transform()[0].outStride, 1);
  ASSERT_EQ(layout.BatchRank(), 1);
  EXPECT_EQ(layout.Batch()[0].inStride, 8);   // real rows are 8 wide
  EXPECT_EQ(layout.Batch()[0].outStride, 5);  // complex rows are 8 / 2 + 1

  const auto [inSize, outSize] = FFTWpp::DataSize<double, Complex>(layout);
  EXPECT_EQ(inSize, 32);
  EXPECT_EQ(outSize, 20);
}

TEST(GuruLayout, ComplexToRealIsTheMirrorOfRealToComplex) {
  const auto layout = FFTWpp::Ranges::ComplexToRealTransformAlong({4, 8}, {1});
  EXPECT_EQ(layout.Batch()[0].inStride, 5);
  EXPECT_EQ(layout.Batch()[0].outStride, 8);
  const auto [inSize, outSize] = FFTWpp::DataSize<Complex, double>(layout);
  EXPECT_EQ(inSize, 20);
  EXPECT_EQ(outSize, 32);
}

//---------------------------------------------------------------------//
//                            Validation                               //
//---------------------------------------------------------------------//

TEST(GuruLayout, RejectsMalformedDimensions) {
  EXPECT_THROW(FFTWpp::Ranges::GuruLayout(std::vector<Dim>{}),
               std::invalid_argument);
  EXPECT_THROW(
      FFTWpp::Ranges::GuruLayout({{.n = 0, .inStride = 1, .outStride = 1}}),
      std::invalid_argument);
  EXPECT_THROW(
      FFTWpp::Ranges::GuruLayout({{.n = 4, .inStride = 0, .outStride = 1}}),
      std::invalid_argument);
}

TEST(GuruLayout, RejectsAnAxisListThatDoesNotSelectAxes) {
  EXPECT_THROW(static_cast<void>(FFTWpp::Ranges::TransformAlong({2, 3}, {})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(FFTWpp::Ranges::TransformAlong({2, 3}, {2})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(FFTWpp::Ranges::TransformAlong({2, 3}, {-1})),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(FFTWpp::Ranges::TransformAlong({2, 3}, {0, 0})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(FFTWpp::Ranges::TransformAlong({2, 3}, {0, 1, 0})),
      std::invalid_argument);
}

TEST(GuruLayout, DetectsOverlappingDimensions) {
  // Two dimensions of extent two both stepping by one: indices 0, 1 and 1, 2
  // overlap at 1.
  const auto overlapping =
      FFTWpp::Ranges::GuruLayout({{.n = 2, .inStride = 1, .outStride = 1}},
                                 {{.n = 2, .inStride = 1, .outStride = 1}});
  EXPECT_THROW(overlapping.ValidateNonOverlapping(FFTWpp::Ranges::Side::Input),
               std::invalid_argument);

  // The same extents nested properly do not overlap.
  const auto nested =
      FFTWpp::Ranges::GuruLayout({{.n = 2, .inStride = 1, .outStride = 1}},
                                 {{.n = 2, .inStride = 2, .outStride = 2}});
  EXPECT_NO_THROW(nested.ValidateNonOverlapping(FFTWpp::Ranges::Side::Input));
}

TEST(GuruLayout, EveryTransformAlongLayoutIsNonOverlapping) {
  for (const auto& axes : std::vector<std::vector<int>>{
           {0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}, {0, 1, 2}}) {
    const auto layout = FFTWpp::Ranges::TransformAlong({3, 4, 5}, axes);
    EXPECT_NO_THROW(layout.ValidateNonOverlapping(FFTWpp::Ranges::Side::Input));
    EXPECT_NO_THROW(
        layout.ValidateNonOverlapping(FFTWpp::Ranges::Side::Output));
    EXPECT_EQ(layout.Extent(FFTWpp::Ranges::Side::Input), 60);
  }
}

TEST(GuruPlan, RejectsRangesTooSmallForTheLayout) {
  const auto layout = FFTWpp::Ranges::TransformAlong({2, 3, 4}, {1});
  auto in = FFTWpp::vector<Complex>(23);  // one short of 24
  auto out = FFTWpp::vector<Complex>(24);
  EXPECT_THROW(FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Estimate,
                                        FFTWpp::Forward),
               std::invalid_argument);
}

TEST(GuruPlan, RejectsTooManyRealToRealKinds) {
  const auto layout = FFTWpp::Ranges::TransformAlong({4, 4}, {1});
  auto in = FFTWpp::vector<double>(16);
  auto out = FFTWpp::vector<double>(16);
  EXPECT_THROW(FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Estimate,
                                        std::vector{FFTWpp::DHT, FFTWpp::DHT}),
               std::invalid_argument);
}

//---------------------------------------------------------------------//
//              The cross-check that actually matters                  //
//---------------------------------------------------------------------//

TYPED_TEST(Guru, InteriorAxisAgreesWithTheTransformDoneByHand) {
  using Real = TypeParam;
  using Scalar = std::complex<Real>;
  constexpr int n0 = 3;
  constexpr int n1 = 8;
  constexpr int n2 = 5;

  auto in = FFTWpp::vector<Scalar>(n0 * n1 * n2);
  FFTWpp::RandomiseValues(in, std::uint64_t{31});

  // One guru plan over the whole array.
  auto guruOut = FFTWpp::vector<Scalar>(n0 * n1 * n2);
  {
    const auto layout = FFTWpp::Ranges::TransformAlong({n0, n1, n2}, {1});
    auto plan = FFTWpp::Ranges::GuruPlan(in, guruOut, layout, FFTWpp::Estimate,
                                         FFTWpp::Forward);
    plan.Execute();
    EXPECT_EQ(plan.Normalisation(), Scalar{1} / Scalar{n1});
  }

  // The same thing the long way: for each i0, a batched rank-1 transform of
  // n2 lines with stride n2 and distance 1. This is as far as the advanced
  // interface reaches, and needs n0 executions rather than one plan.
  auto referenceOut = FFTWpp::vector<Scalar>(n0 * n1 * n2);
  {
    const auto slab =
        FFTWpp::Ranges::Layout(1, std::vector{n1}, n2, std::vector{n1}, n2, 1);
    for (int i0 = 0; i0 < n0; ++i0) {
      auto inSlab = FFTWpp::vector<Scalar>(n1 * n2);
      auto outSlab = FFTWpp::vector<Scalar>(n1 * n2);
      std::copy_n(in.begin() + i0 * n1 * n2, n1 * n2, inSlab.begin());
      auto plan = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(inSlab, slab),
                                       FFTWpp::Ranges::View(outSlab, slab),
                                       FFTWpp::Estimate, FFTWpp::Forward);
      plan.Execute();
      std::copy_n(outSlab.begin(), n1 * n2,
                  referenceOut.begin() + i0 * n1 * n2);
    }
  }

  EXPECT_TRUE(FFTWpp::CheckValues(referenceOut, guruOut, Scalar{1}));
}

//---------------------------------------------------------------------//
//                           Round trips                               //
//---------------------------------------------------------------------//

TYPED_TEST(Guru, ComplexRoundTripAlongEachAxisOfAThreeDimensionalArray) {
  using Scalar = std::complex<TypeParam>;
  constexpr int n0 = 3;
  constexpr int n1 = 4;
  constexpr int n2 = 5;

  for (int axis = 0; axis < 3; ++axis) {
    const auto layout = FFTWpp::Ranges::TransformAlong({n0, n1, n2}, {axis});
    auto in = FFTWpp::vector<Scalar>(n0 * n1 * n2);
    auto out = FFTWpp::vector<Scalar>(n0 * n1 * n2);
    auto back = FFTWpp::vector<Scalar>(n0 * n1 * n2);

    auto forward = FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Measure,
                                            FFTWpp::Forward);
    auto backward = FFTWpp::Ranges::GuruPlan(out, back, layout, FFTWpp::Measure,
                                             FFTWpp::Backward);
    FFTWpp::RandomiseValues(in, static_cast<std::uint64_t>(40 + axis));
    const auto original = in;
    forward.Execute();
    backward.Execute();

    EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()))
        << "axis " << axis;
  }
}

TYPED_TEST(Guru, RealToComplexRoundTripAlongAnInteriorAxis) {
  using Real = TypeParam;
  using Scalar = std::complex<Real>;
  const auto shape = std::vector<std::ptrdiff_t>{3, 8, 5};
  const auto axes = std::vector<int>{1};

  const auto forwardLayout =
      FFTWpp::Ranges::RealToComplexTransformAlong(shape, axes);
  const auto backwardLayout =
      FFTWpp::Ranges::ComplexToRealTransformAlong(shape, axes);

  const auto [realSize, complexSize] =
      FFTWpp::DataSize<Real, Scalar>(forwardLayout);
  EXPECT_EQ(realSize, 3 * 8 * 5);
  EXPECT_EQ(complexSize, 3 * 5 * 5);

  auto in = FFTWpp::vector<Real>(realSize);
  auto out = FFTWpp::vector<Scalar>(complexSize);
  auto back = FFTWpp::vector<Real>(realSize);

  auto forward =
      FFTWpp::Ranges::GuruPlan(in, out, forwardLayout, FFTWpp::Measure);
  auto backward =
      FFTWpp::Ranges::GuruPlan(out, back, backwardLayout, FFTWpp::Measure);

  FFTWpp::RandomiseValues(in, std::uint64_t{41});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(forward.Normalisation(), backward.Normalisation());
  EXPECT_EQ(backward.Normalisation(), Real{1} / Real{8});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Guru, RealToRealRoundTripAlongAnInteriorAxis) {
  using Real = TypeParam;
  const auto layout = FFTWpp::Ranges::TransformAlong({3, 8, 5}, {1});

  auto in = FFTWpp::vector<Real>(3 * 8 * 5);
  auto out = FFTWpp::vector<Real>(3 * 8 * 5);
  auto back = FFTWpp::vector<Real>(3 * 8 * 5);

  auto forward = FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Measure,
                                          FFTWpp::REDFT10);
  auto backward = FFTWpp::Ranges::GuruPlan(out, back, layout, FFTWpp::Measure,
                                           FFTWpp::REDFT01);

  FFTWpp::RandomiseValues(in, std::uint64_t{42});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(backward.Normalisation(), Real{1} / Real{16});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Guru, TransposingLayoutsAreExpressible) {
  // A rank-1 transform along the rows of a row-major array, written to a
  // column-major output. The two sides have different stride orders, which
  // the advanced interface cannot describe at all.
  using Scalar = std::complex<TypeParam>;
  constexpr int rows = 4;
  constexpr int columns = 6;

  const auto layout = FFTWpp::Ranges::GuruLayout(
      {{.n = columns, .inStride = 1, .outStride = rows}},
      {{.n = rows, .inStride = columns, .outStride = 1}});

  auto in = FFTWpp::vector<Scalar>(rows * columns);
  auto transposed = FFTWpp::vector<Scalar>(rows * columns);
  auto plan = FFTWpp::Ranges::GuruPlan(in, transposed, layout, FFTWpp::Estimate,
                                       FFTWpp::Forward);
  FFTWpp::RandomiseValues(in, std::uint64_t{43});

  // The same transform without the transposition, for comparison.
  auto straight = FFTWpp::vector<Scalar>(rows * columns);
  {
    const auto plain = FFTWpp::Ranges::Layout(1, std::vector{columns}, rows,
                                              std::vector{columns}, 1, columns);
    auto reference = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, plain),
                                          FFTWpp::Ranges::View(straight, plain),
                                          FFTWpp::Estimate, FFTWpp::Forward);
    reference.Execute();
  }
  plan.Execute();

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      EXPECT_EQ(transposed[c * rows + r], straight[r * columns + c])
          << "row " << r << ", column " << c;
    }
  }
}

//---------------------------------------------------------------------//
//                       Ownership and execution                       //
//---------------------------------------------------------------------//

TEST(GuruPlan, OwnershipMatchesRangesPlan) {
  const auto before = FFTWpp::LivePlanCount();
  const auto layout = FFTWpp::Ranges::TransformAlong({4, 4}, {0});
  auto in = FFTWpp::vector<Complex>(16);
  auto out = FFTWpp::vector<Complex>(16);

  auto source = FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Estimate,
                                         FFTWpp::Forward);
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);

  {
    auto copy = source;
    EXPECT_NE(copy.Pointer(), source.Pointer());
    EXPECT_EQ(FFTWpp::LivePlanCount(), before + 2);
  }
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);

  const auto pointer = source.Pointer();
  auto moved = std::move(source);
  EXPECT_EQ(moved.Pointer(), pointer);
  EXPECT_TRUE(source.IsNull());
  EXPECT_EQ(FFTWpp::LivePlanCount(), before + 1);
}

TEST(GuruPlan, TransformsTheCallersBufferRatherThanACopyOfIt) {
  // A regression test with teeth. GuruPlan takes views, not containers, so
  // that the deduction guide is the only viable one: if a container could be
  // deduced it would be passed by value, the plan would be built on the copy's
  // storage, and the caller's output would stay untouched. A round trip does
  // not notice, because a copy round-trips perfectly well.
  const auto layout =
      FFTWpp::Ranges::GuruLayout({{.n = 4, .inStride = 1, .outStride = 1}});
  auto in = FFTWpp::vector<Complex>(4);
  auto out = FFTWpp::vector<Complex>(4);
  for (int i = 0; i < 4; ++i) in[i] = Complex(i + 1, 0);

  auto plan = FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Estimate,
                                       FFTWpp::Forward);
  plan.Execute();

  // The zero-frequency term is the sum of the input, which is 1 + 2 + 3 + 4.
  EXPECT_EQ(out[0], Complex(10, 0));
  EXPECT_TRUE(std::ranges::any_of(out, [](auto z) { return z != Complex{}; }));
}

TEST(GuruPlan, NewArrayExecutionIsChecked) {
  const auto layout = FFTWpp::Ranges::TransformAlong({4, 4}, {0});
  auto in = FFTWpp::vector<Complex>(16);
  auto out = FFTWpp::vector<Complex>(16);
  auto plan = FFTWpp::Ranges::GuruPlan(in, out, layout, FFTWpp::Estimate,
                                       FFTWpp::Forward);

  auto otherIn = FFTWpp::vector<Complex>(16);
  auto otherOut = FFTWpp::vector<Complex>(16);
  EXPECT_TRUE(plan.CanExecuteOn(otherIn, otherOut));
  EXPECT_NO_THROW(plan.ExecuteChecked(otherIn, otherOut));

  auto tooSmall = FFTWpp::vector<Complex>(8);
  EXPECT_FALSE(plan.CanExecuteOn(tooSmall, otherOut));
  EXPECT_THROW(plan.ExecuteChecked(tooSmall, otherOut), std::invalid_argument);
}

TEST(GuruPlan, TheSixtyFourBitEntryPointIsChosenByMagnitude) {
  // Only the predicate is exercised; planning a transform that needs the
  // 64-bit interface would need an array larger than memory.
  const auto small = FFTWpp::Dim{.n = 8, .inStride = 1, .outStride = 1};
  EXPECT_TRUE(FFTWpp::Internal::FitsGuru32(&small, 1));

  const auto large = FFTWpp::Dim{
      .n = static_cast<std::ptrdiff_t>(std::numeric_limits<int>::max()) + 1,
      .inStride = 1,
      .outStride = 1};
  EXPECT_FALSE(FFTWpp::Internal::FitsGuru32(&large, 1));
  EXPECT_TRUE(FFTWpp::Internal::FitsGuru32(&large, 0));
}

}  // namespace
