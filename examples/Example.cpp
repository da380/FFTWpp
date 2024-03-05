#include <FFTWpp/Core>
#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

int main() {
  using namespace FFTWpp;

  // Set the precision.
  using Real = double;
  using Complex = std::complex<Real>;

  // Set the data types.
  using InType = Real;
  using OutType = Complex;

  // Generate allocate input and output arrays.
  auto n = 64;
  auto [inSize, outSize] = FFTWpp::DataSize<InType, OutType>(n);
  auto in = FFTWpp::vector<InType>(inSize);
  auto out = FFTWpp::vector<OutType>(outSize);
  auto copy = FFTWpp::vector<InType>(inSize);

  // Raw fftw3 section.
  {
    // Make the plans.
    auto planForward = fftw_plan_dft_r2c_1d(
        n, in.data(), reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_MEASURE);
    auto planBackward =
        fftw_plan_dft_c2r_1d(n, reinterpret_cast<fftw_complex*>(out.data()),
                             copy.data(), FFTW_MEASURE);
    // Set in values.
    FFTWpp::RandomizeValues(in);
    // Execute plans.
    fftw_execute(planForward);
    fftw_execute(planBackward);
    // Normalise the inverse transformation.
    std::transform(copy.begin(), copy.end(), copy.begin(),
                   [n](auto x) { return x / static_cast<double>(n); });
    // Print the error on the transform pair.
    std::cout << "Raw fftw3 section:\n Error = "
              << std::ranges::max(std::ranges::views::zip_transform(
                     [](auto x, auto y) { return std::abs(x - y); },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
    // Delete the plans to free memory.
    fftw_destroy_plan(planForward);
    fftw_destroy_plan(planBackward);
  }

  // Transforms done using FFTWpp/Core.
  {
    // Make the plans.
    auto planForward = Plan(n, in.data(), out.data(), FFTW_MEASURE);
    auto planBackward = Plan(n, out.data(), copy.data(), FFTW_MEASURE);
    // Set in values.
    FFTWpp::RandomizeValues(in);
    // Execute the plans.
    Execute(planForward);
    Execute(planBackward);
    // Normalise the inverse transformation.
    std::transform(copy.begin(), copy.end(), copy.begin(),
                   [n](auto x) { return x / static_cast<double>(n); });
    // Print the error on the transform pair.
    std::cout << "Minimal FFTWpp section:\n Error = "
              << std::ranges::max(std::ranges::views::zip_transform(
                     [](auto x, auto y) { return std::abs(x - y); },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
    // Delete the plans to free memory.
    Destroy(planForward);
    Destroy(planBackward);
  }

  // Full usage of FFTWpp
  {
    auto inLayout =
        Ranges::Layout(1, std::vector{inSize}, 1, std::vector{inSize}, 1, 1);
    auto outLayout =
        Ranges::Layout(1, std::vector{outSize}, 1, std::vector{outSize}, 1, 1);
    // Make the plans.
    auto planForward = Ranges::Plan(Ranges::View(in, inLayout),
                                    Ranges::View(out, outLayout), Measure);
    // auto planBackward = Ranges::Plan(Ranges::View(out, outLayout),
    // Ranges::View(copy, inLayout), Measure);
    // Set in values.
    // FFTWpp::RandomizeValues(in);
    in = FFTWpp::vector<InType>(inSize, 1);
    // Execute the plans.
    planForward.Execute();
    // planBackward.Execute();

    for (auto val : out) std::cout << val << std::endl;

    /*
      // Normalise the inverse transformation.
      std::transform(copy.begin(), copy.end(), copy.begin(),
                     [&](auto x) { return x / static_cast<Real>(n); });
      // Print the error on the transform pair.
      std::cout << "Full FFTWpp section:\n Error = "
                << std::ranges::max(std::ranges::views::zip_transform(
                       [](auto x, auto y) { return std::abs(x - y); },
                       std::ranges::views::all(in),
                       std::ranges::views::all(copy)))
                << std::endl;
                */
  }
}
