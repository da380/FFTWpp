#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <typeinfo>
#include <vector>

int main() {
  using Real = double;
  using Complex = std::complex<Real>;
  using RealVector = FFTWpp::vector<Real>;
  using ComplexVector = FFTWpp::vector<Complex>;

  {
    // set the size of
    int n = 10;

    // Initialise the vectors.
    RealVector in(n), check(n);
    ComplexVector out(n / 2 + 1);

    // Set the input values.
    for (auto& x : in) x = 1;

    // Form the data layouts.
    auto inLayout =
        FFTWpp::DataLayout(1, std::vector{n}, 1, std::vector{n}, 1, 1);

    auto outLayout = FFTWpp::DataLayout(1, std::vector{n / 2 + 1}, 1,
                                        std::vector{n / 2 + 1}, 1, 1);

    // Generate wisdom.
    FFTWpp::GenerateWisdom<Real, Complex, true>(inLayout, outLayout,
                                                FFTWpp::Exhaustive);

    // Form the data views.
    auto inView = FFTWpp::DataView(in.begin(), in.end(), inLayout);
    auto outView = FFTWpp::DataView(out.begin(), out.end(), outLayout);
    auto checkView = FFTWpp::DataView(check.begin(), check.end(), inLayout);

    // Form the plans using the wisdom generated.
    auto forward_plan = FFTWpp::Plan(inView, outView, FFTWpp::WisdomOnly);
    auto backward_plan = FFTWpp::Plan(outView, checkView, FFTWpp::WisdomOnly);

    // Do some moves for fun.
    auto copy_plan = std::move(forward_plan);
    forward_plan = std::move(copy_plan);

    // Execute the plans.
    forward_plan.Execute();
    backward_plan.Execute();

    // Normalise the results.
    auto norm = backward_plan.Normalisation();
    std::transform(check.begin(), check.end(), check.begin(),
                   [norm](auto x) { return norm * x; });

    // Print in and check for comparison.
    for (auto& x : in) std::cout << x << std::endl;
    std::cout << std::string(10, '=') << std::endl;
    for (auto& x : check) std::cout << x << std::endl;
  }

  FFTWpp::CleanUp();
}
