
#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

int main() {
  using namespace FFTWpp;

  using Real = double;
  using Complex = std::complex<Real>;

  // 2D complex-complex transformation. Data are stored in
  // the default row-major ordering.
  {
    // Set dimension.
    int n0 = 10;
    int n1 = 20;
    auto [sizeIn, sizeOut] = FFTWpp::DataSize<Complex, Complex>(n0, n1);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Complex>(sizeIn);
    auto out = FFTWpp::vector<Complex>(sizeOut);
    auto copy = FFTWpp::vector<Complex>(sizeIn);

    // Form the plans
    auto planForward = Ranges::Plan(
        Ranges::View(in, n0, n1), Ranges::View(out, n0, n1), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out, n0, n1), Ranges::View(copy, n0, n1),
                     Measure, Backward);

    // Set values for in.
    FFTWpp::RandomiseValues(in);

    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();

    // Print the error on the transform pair.
    std::cout << std::ranges::max(std::ranges::views::zip_transform(
                     [&planBackward](auto x, auto y) {
                       return std::abs(x - y * planBackward.Normalisation());
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
  }

  // 4D complex-complex transformation. Data are stored in
  // the default row-major ordering.
  {
    // Set dimension.
    int n0 = 10;
    int n1 = 20;
    int n2 = 4;
    int n3 = 5;
    auto [sizeIn, sizeOut] = FFTWpp::DataSize<Complex, Complex>(n0, n1, n2, n3);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Complex>(sizeIn);
    auto out = FFTWpp::vector<Complex>(sizeOut);
    auto copy = FFTWpp::vector<Complex>(sizeIn);

    // Form the plans
    auto planForward =
        Ranges::Plan(Ranges::View(in, n0, n1, n2, n3),
                     Ranges::View(out, n0, n1, n2, n3), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out, n0, n1, n2, n3),
                     Ranges::View(copy, n0, n1, n2, n3), Measure, Backward);

    // Set values for in.
    FFTWpp::RandomiseValues(in);

    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();

    // Print the error on the transform pair.
    std::cout << std::ranges::max(std::ranges::views::zip_transform(
                     [&planBackward](auto x, auto y) {
                       return std::abs(x - y * planBackward.Normalisation());
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
  }
}