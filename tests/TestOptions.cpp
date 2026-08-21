// Tests for the type-safe option wrappers, the utility helpers and the
// wisdom routes that have no coverage elsewhere.
#include <gtest/gtest.h>

#include <FFTWpp/Ranges>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <type_traits>

namespace {

using Complex = std::complex<double>;

//---------------------------------------------------------------------//
//                              Options                                //
//---------------------------------------------------------------------//

TEST(Options, DirectionsCarryTheirFftwValues) {
  static_assert(static_cast<int>(FFTWpp::Forward) == FFTW_FORWARD);
  static_assert(static_cast<int>(FFTWpp::Backward) == FFTW_BACKWARD);
  static_assert(FFTWpp::Forward != FFTWpp::Backward);
  static_assert(FFTWpp::Direction{} == FFTWpp::Forward);
}

TEST(Options, FlagsCombineWithBitwiseOr) {
  constexpr auto combined = FFTWpp::Measure | FFTWpp::Unaligned;
  static_assert(static_cast<unsigned>(combined) ==
                (FFTW_MEASURE | FFTW_UNALIGNED));

  auto accumulated = FFTWpp::Flag{FFTW_MEASURE};
  accumulated |= FFTWpp::Unaligned;
  EXPECT_EQ(static_cast<unsigned>(accumulated),
            static_cast<unsigned>(combined));

  // Three flags at a time exercises the operator on a temporary as well as on
  // the named constants.
  constexpr auto three =
      FFTWpp::Measure | FFTWpp::Unaligned | FFTWpp::DestroyInput;
  static_assert(static_cast<unsigned>(three) ==
                (FFTW_MEASURE | FFTW_UNALIGNED | FFTW_DESTROY_INPUT));
}

TEST(Options, EveryRealKindInvertsBackToItself) {
  constexpr FFTWpp::RealKind kinds[] = {
      FFTWpp::R2HC,    FFTWpp::HC2R,    FFTWpp::DHT,     FFTWpp::REDFT00,
      FFTWpp::REDFT10, FFTWpp::REDFT01, FFTWpp::REDFT11, FFTWpp::RODFT00,
      FFTWpp::RODFT10, FFTWpp::RODFT01, FFTWpp::RODFT11};
  for (auto kind : kinds) {
    EXPECT_EQ(kind.Inverse().Inverse(), kind);
  }
}

TEST(Options, LogicalDimensionsMatchTheFftwDefinitions) {
  constexpr int n = 8;
  EXPECT_EQ(FFTWpp::R2HC.LogicalDimension(n), n);
  EXPECT_EQ(FFTWpp::DHT.LogicalDimension(n), n);
  EXPECT_EQ(FFTWpp::REDFT00.LogicalDimension(n), 2 * (n - 1));
  EXPECT_EQ(FFTWpp::RODFT00.LogicalDimension(n), 2 * (n + 1));
  EXPECT_EQ(FFTWpp::REDFT10.LogicalDimension(n), 2 * n);
  EXPECT_EQ(FFTWpp::RODFT11.LogicalDimension(n), 2 * n);
}

//---------------------------------------------------------------------//
//                              Utility                                //
//---------------------------------------------------------------------//

TEST(Utility, DataSizeIsAConstantExpression) {
  static_assert(FFTWpp::DataSize<Complex, Complex>(8).first == 8);
  static_assert(FFTWpp::DataSize<Complex, Complex>(8).second == 8);
  static_assert(FFTWpp::DataSize<double, Complex>(8).first == 8);
  static_assert(FFTWpp::DataSize<double, Complex>(8).second == 5);
  static_assert(FFTWpp::DataSize<Complex, double>(8).first == 5);
  static_assert(FFTWpp::DataSize<Complex, double>(8).second == 8);
  static_assert(FFTWpp::DataSize<double, Complex>(4, 6).second == 4 * 4);
}

TEST(Utility, RandomiseValuesIsReproducibleFromASeed) {
  auto first = FFTWpp::vector<Complex>(64);
  auto second = FFTWpp::vector<Complex>(64);
  FFTWpp::RandomiseValues(first, std::uint64_t{12345});
  FFTWpp::RandomiseValues(second, std::uint64_t{12345});
  EXPECT_TRUE(std::ranges::equal(first, second));

  auto third = FFTWpp::vector<Complex>(64);
  FFTWpp::RandomiseValues(third, std::uint64_t{54321});
  EXPECT_FALSE(std::ranges::equal(first, third));
}

TEST(Utility, RandomiseValuesFillsBothComplexComponents) {
  auto values = FFTWpp::vector<Complex>(256);
  FFTWpp::RandomiseValues(values, std::uint64_t{7});
  EXPECT_TRUE(std::ranges::any_of(
      values, [](auto z) { return z.real() != 0 && z.imag() != 0; }));
  EXPECT_TRUE(std::ranges::none_of(
      values, [](auto z) { return z.real() == z.imag(); }));
}

TEST(Utility, CheckValuesComparesRangesOfDifferentTypes) {
  auto data = FFTWpp::vector<double>{1, 2, 3, 4};
  auto scaled = std::vector<double>{2, 4, 6, 8};
  EXPECT_TRUE(FFTWpp::CheckValues(data, scaled, 0.5));
  EXPECT_FALSE(FFTWpp::CheckValues(data, scaled, 1.0));
}

TEST(Utility, CheckValuesHonoursAnExplicitTolerance) {
  auto data = FFTWpp::vector<double>{1.0};
  auto perturbed = FFTWpp::vector<double>{1.0 + 1e-6};
  EXPECT_FALSE(FFTWpp::CheckValues(data, perturbed, 1.0));
  EXPECT_TRUE(FFTWpp::CheckValues(data, perturbed, 1.0, 1e-5));
}

//---------------------------------------------------------------------//
//                               Wisdom                                //
//---------------------------------------------------------------------//

TEST(Wisdom, RoundTripsThroughAString) {
  const auto layout = FFTWpp::Ranges::Layout(16);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Measure);

  const auto serialised = FFTWpp::ExportWisdomToString<double>();
  EXPECT_FALSE(serialised.empty());

  FFTWpp::ForgetWisdom();
  auto in = FFTWpp::vector<Complex>(16);
  auto out = FFTWpp::vector<Complex>(16);
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::WisdomOnly, FFTWpp::Forward),
      std::runtime_error);

  FFTWpp::ImportWisdomFromString<double>(serialised);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                       FFTWpp::Ranges::View(out),
                                       FFTWpp::WisdomOnly, FFTWpp::Forward));
}

TEST(Wisdom, EachPrecisionHasItsOwnStore) {
  const auto layout = FFTWpp::Ranges::Layout(16);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<std::complex<float>, std::complex<float>>(
      layout, layout, FFTWpp::Measure);

  // Single-precision wisdom exists; double-precision wisdom does not.
  auto singleIn = FFTWpp::vector<std::complex<float>>(16);
  auto singleOut = FFTWpp::vector<std::complex<float>>(16);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(singleIn),
                                       FFTWpp::Ranges::View(singleOut),
                                       FFTWpp::WisdomOnly, FFTWpp::Forward));

  auto doubleIn = FFTWpp::vector<Complex>(16);
  auto doubleOut = FFTWpp::vector<Complex>(16);
  EXPECT_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(doubleIn),
                                    FFTWpp::Ranges::View(doubleOut),
                                    FFTWpp::WisdomOnly, FFTWpp::Forward),
               std::runtime_error);

  // Clearing one precision leaves the others alone.
  FFTWpp::ForgetWisdom<double>();
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(singleIn),
                                       FFTWpp::Ranges::View(singleOut),
                                       FFTWpp::WisdomOnly, FFTWpp::Forward));
  FFTWpp::ForgetWisdom<float>();
  EXPECT_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(singleIn),
                                    FFTWpp::Ranges::View(singleOut),
                                    FFTWpp::WisdomOnly, FFTWpp::Forward),
               std::runtime_error);
}

TEST(Wisdom, SinglePrecisionFilesRoundTrip) {
  const auto path =
      std::filesystem::temp_directory_path() / "fftwpp-test-wisdom-float.dat";
  const auto layout = FFTWpp::Ranges::Layout(16);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<std::complex<float>, std::complex<float>>(
      layout, layout, FFTWpp::Measure);

  ASSERT_NO_THROW(FFTWpp::ExportWisdom<float>(path.string()));
  FFTWpp::ForgetWisdom();
  EXPECT_NO_THROW(FFTWpp::ImportWisdom<float>(path.string()));

  auto in = FFTWpp::vector<std::complex<float>>(16);
  auto out = FFTWpp::vector<std::complex<float>>(16);
  EXPECT_NO_THROW(FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in),
                                       FFTWpp::Ranges::View(out),
                                       FFTWpp::WisdomOnly, FFTWpp::Forward));
  EXPECT_TRUE(std::filesystem::remove(path));
}

TEST(Wisdom, ReportsTheFileItCouldNotRead) {
  const auto path = std::filesystem::temp_directory_path() /
                    "fftwpp-test-wisdom-does-not-exist.dat";
  std::filesystem::remove(path);
  try {
    FFTWpp::ImportWisdom<double>(path.string());
    FAIL() << "importing absent wisdom should throw";
  } catch (const std::runtime_error& error) {
    const auto message = std::string(error.what());
    EXPECT_NE(message.find("double"), std::string::npos);
    EXPECT_NE(message.find(path.string()), std::string::npos);
  }
}

TEST(Wisdom, GeneratingWithEstimateIsANoOp) {
  const auto layout = FFTWpp::Ranges::Layout(16);
  FFTWpp::ForgetWisdom();
  FFTWpp::GenerateWisdom<Complex, Complex>(layout, layout, FFTWpp::Estimate);

  auto in = FFTWpp::vector<Complex>(16);
  auto out = FFTWpp::vector<Complex>(16);
  EXPECT_THROW(
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::WisdomOnly, FFTWpp::Forward),
      std::runtime_error);
}

}  // namespace
