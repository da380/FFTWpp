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
  using Float = double;
  using Complex = std::complex<Float>;
  using RealVector = FFTWpp::vector<Float>;
  using ComplexVector = FFTWpp::vector<Complex>;

  {
    // generate a random size for the data
    int n = 10;

    // Initialise the vectors.
    RealVector in(n), check(n);
    ComplexVector out(n / 2 + 1);

    // Set the input values.
    for (auto& x : in) x = 1;

    // Form the data views
    auto inView = FFTWpp::MakeDataView1D(in);
    auto outView = FFTWpp::MakeDataView1D(out);
    auto checkView = FFTWpp::MakeDataView1D(check);

    auto forward_plan =
        FFTWpp::Plan(inView, outView, FFTWpp::Measure, FFTWpp::Forward);

    auto backward_plan =
        FFTWpp::Plan(outView, checkView, FFTWpp::Measure, FFTWpp::Backward);

    forward_plan.Execute();

    backward_plan.Execute();

    auto norm = backward_plan.Normalisation();
    std::transform(check.begin(), check.end(), check.begin(),
                   [norm](auto x) { return norm * x; });

    for (auto& x : in) std::cout << x << std::endl;
    std::cout << std::string(10, '=') << std::endl;

    for (auto& x : check) std::cout << x << std::endl;
  }

  FFTWpp::CleanUp();
}
