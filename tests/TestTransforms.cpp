// Round-trip correctness for the multi-dimensional and batched transforms
// that the advanced FFTW interface exists for. Ranges::Layout, Ranges::View
// and Ranges::Plan are the part of this library that consumers lean on
// hardest, and the batched shapes are the ones that were previously covered
// only by the examples.
#include <gtest/gtest.h>

#include <FFTWpp/Ranges>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

template <typename Real>
class Transforms : public testing::Test {};

using Precisions = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(Transforms, Precisions);

//---------------------------------------------------------------------//
//                       Multi-dimensional                             //
//---------------------------------------------------------------------//

TYPED_TEST(Transforms, TwoDimensionalComplexRoundTrip) {
  using Complex = std::complex<TypeParam>;
  constexpr int n0 = 12;
  constexpr int n1 = 20;

  auto in = FFTWpp::vector<Complex>(n0 * n1);
  auto out = FFTWpp::vector<Complex>(n0 * n1);
  auto back = FFTWpp::vector<Complex>(n0 * n1);

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, n0, n1),
                                      FFTWpp::Ranges::View(out, n0, n1),
                                      FFTWpp::Measure, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out, n0, n1),
                                       FFTWpp::Ranges::View(back, n0, n1),
                                       FFTWpp::Measure, FFTWpp::Backward);

  FFTWpp::RandomiseValues(in, std::uint64_t{1});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(backward.Normalisation(), Complex{1} / Complex{n0 * n1});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Transforms, ThreeDimensionalRealToComplexRoundTrip) {
  using Real = TypeParam;
  using Complex = std::complex<Real>;
  constexpr int n0 = 6;
  constexpr int n1 = 8;
  constexpr int n2 = 10;

  const auto [inSize, outSize] = FFTWpp::DataSize<Real, Complex>(n0, n1, n2);
  EXPECT_EQ(inSize, n0 * n1 * n2);
  EXPECT_EQ(outSize, n0 * n1 * (n2 / 2 + 1));

  auto in = FFTWpp::vector<Real>(inSize);
  auto out = FFTWpp::vector<Complex>(outSize);
  auto back = FFTWpp::vector<Real>(inSize);

  auto forward = FFTWpp::Ranges::Plan(
      FFTWpp::Ranges::View(in, n0, n1, n2),
      FFTWpp::Ranges::View(out, n0, n1, n2 / 2 + 1), FFTWpp::Measure);
  auto backward = FFTWpp::Ranges::Plan(
      FFTWpp::Ranges::View(out, n0, n1, n2 / 2 + 1),
      FFTWpp::Ranges::View(back, n0, n1, n2), FFTWpp::Measure);

  FFTWpp::RandomiseValues(in, std::uint64_t{2});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  // Both directions describe the same logical transform, so both must report
  // the same normalisation. Before Normalisation() was corrected the forward
  // plan reported 1 / (n0 * n1 * (n2 / 2 + 1)).
  EXPECT_EQ(forward.Normalisation(), backward.Normalisation());
  EXPECT_EQ(backward.Normalisation(), Real{1} / Real{n0 * n1 * n2});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Transforms, TwoDimensionalRealToRealRoundTripPerDimensionKinds) {
  using Real = TypeParam;
  constexpr int n0 = 8;
  constexpr int n1 = 12;

  auto in = FFTWpp::vector<Real>(n0 * n1);
  auto out = FFTWpp::vector<Real>(n0 * n1);
  auto back = FFTWpp::vector<Real>(n0 * n1);

  auto forward = FFTWpp::Ranges::Plan(
      FFTWpp::Ranges::View(in, n0, n1), FFTWpp::Ranges::View(out, n0, n1),
      FFTWpp::Measure, FFTWpp::REDFT10, FFTWpp::RODFT10);
  auto backward = FFTWpp::Ranges::Plan(
      FFTWpp::Ranges::View(out, n0, n1), FFTWpp::Ranges::View(back, n0, n1),
      FFTWpp::Measure, FFTWpp::REDFT01, FFTWpp::RODFT01);

  FFTWpp::RandomiseValues(in, std::uint64_t{3});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(backward.Normalisation(), Real{1} / Real{4 * n0 * n1});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Transforms, ASingleRealToRealKindAppliesToEveryDimension) {
  using Real = TypeParam;
  constexpr int n0 = 8;
  constexpr int n1 = 8;

  auto in = FFTWpp::vector<Real>(n0 * n1);
  auto out = FFTWpp::vector<Real>(n0 * n1);
  auto back = FFTWpp::vector<Real>(n0 * n1);

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, n0, n1),
                                      FFTWpp::Ranges::View(out, n0, n1),
                                      FFTWpp::Measure, FFTWpp::DHT);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out, n0, n1),
                                       FFTWpp::Ranges::View(back, n0, n1),
                                       FFTWpp::Measure, FFTWpp::DHT);

  FFTWpp::RandomiseValues(in, std::uint64_t{5});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

//---------------------------------------------------------------------//
//                        Batched transforms                           //
//---------------------------------------------------------------------//

TYPED_TEST(Transforms, BatchedContiguousComplexRoundTrip) {
  // howMany transforms, each in its own contiguous block: stride 1, dist n.
  using Complex = std::complex<TypeParam>;
  constexpr int n = 24;
  constexpr int howMany = 7;

  const auto layout =
      FFTWpp::Ranges::Layout(1, std::vector{n}, howMany, std::vector{n}, 1, n);
  ASSERT_EQ(layout.size(), n * howMany);

  auto in = FFTWpp::vector<Complex>(layout.size());
  auto out = FFTWpp::vector<Complex>(layout.size());
  auto back = FFTWpp::vector<Complex>(layout.size());

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, layout),
                                      FFTWpp::Ranges::View(out, layout),
                                      FFTWpp::Measure, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out, layout),
                                       FFTWpp::Ranges::View(back, layout),
                                       FFTWpp::Measure, FFTWpp::Backward);

  FFTWpp::RandomiseValues(in, std::uint64_t{6});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(backward.Normalisation(), Complex{1} / Complex{n});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Transforms, BatchedInterleavedComplexRoundTrip) {
  // The interleaved layout: element i of transform j lives at j + howMany * i,
  // so stride is howMany and dist is 1. This is the descriptor that lets a
  // consumer transform a tensor in place rather than repacking it, and is the
  // reason the advanced interface is worth wrapping at all.
  using Complex = std::complex<TypeParam>;
  constexpr int n = 16;
  constexpr int howMany = 5;

  const auto layout = FFTWpp::Ranges::Layout(1, std::vector{n}, howMany,
                                             std::vector{n}, howMany, 1);
  auto in = FFTWpp::vector<Complex>(n * howMany);
  auto out = FFTWpp::vector<Complex>(n * howMany);
  auto back = FFTWpp::vector<Complex>(n * howMany);

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, layout),
                                      FFTWpp::Ranges::View(out, layout),
                                      FFTWpp::Measure, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out, layout),
                                       FFTWpp::Ranges::View(back, layout),
                                       FFTWpp::Measure, FFTWpp::Backward);

  FFTWpp::RandomiseValues(in, std::uint64_t{7});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

TYPED_TEST(Transforms, BatchedRealToComplexRoundTrip) {
  using Real = TypeParam;
  using Complex = std::complex<Real>;
  constexpr int n = 20;
  constexpr int m = n / 2 + 1;
  constexpr int howMany = 4;

  const auto realLayout =
      FFTWpp::Ranges::Layout(1, std::vector{n}, howMany, std::vector{n}, 1, n);
  const auto complexLayout =
      FFTWpp::Ranges::Layout(1, std::vector{m}, howMany, std::vector{m}, 1, m);

  auto in = FFTWpp::vector<Real>(realLayout.size());
  auto out = FFTWpp::vector<Complex>(complexLayout.size());
  auto back = FFTWpp::vector<Real>(realLayout.size());

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in, realLayout),
                                      FFTWpp::Ranges::View(out, complexLayout),
                                      FFTWpp::Measure);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out, complexLayout),
                                       FFTWpp::Ranges::View(back, realLayout),
                                       FFTWpp::Measure);

  FFTWpp::RandomiseValues(in, std::uint64_t{8});
  const auto original = in;
  forward.Execute();
  backward.Execute();

  EXPECT_EQ(backward.Normalisation(), Real{1} / Real{n});
  EXPECT_TRUE(FFTWpp::CheckValues(original, back, backward.Normalisation()));
}

//---------------------------------------------------------------------//
//                            In place                                 //
//---------------------------------------------------------------------//

TYPED_TEST(Transforms, InPlaceComplexRoundTrip) {
  using Complex = std::complex<TypeParam>;
  constexpr int n = 32;

  auto data = FFTWpp::vector<Complex>(n);
  FFTWpp::RandomiseValues(data, std::uint64_t{9});
  const auto original = data;

  auto forward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(data),
                                      FFTWpp::Ranges::View(data),
                                      FFTWpp::Estimate, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(data),
                                       FFTWpp::Ranges::View(data),
                                       FFTWpp::Estimate, FFTWpp::Backward);
  forward.Execute();
  backward.Execute();

  EXPECT_TRUE(FFTWpp::CheckValues(original, data, backward.Normalisation()));
}

}  // namespace
