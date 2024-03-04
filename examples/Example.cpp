#include <FFTWpp/Core>
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

  // Set the input and output arrays.
  auto n = 5;
  auto in = std::vector<Real>(n);
  auto out = std::vector<Complex>(n / 2 + 1);

  // Raw fftw3
  {
    // For the plans.
    auto plan = fftw_plan_dft_r2c_1d(
        n, in.data(), reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_ESTIMATE);

    // Execute the plans.
    fftw_execute(plan);
  }

  // Minimal usage of FFTWpp requiring inclusions of <FFTWpp/Core>
  {
    auto plan = Plan(n, in.data(), out.data(), FFTW_ESTIMATE);
    Execute(plan);
  }

  // Full usage of FFTWpp requiring inclusion of <FFTWpp/Ranges>
  {
    auto plan = Ranges::Plan(Ranges::View(in), Ranges::View(out), Estimate);
    plan.Execute();
    auto plan2 = std::move(plan);
    plan2.Execute();
  }

  {
    auto inLayout = Ranges::Layout(2, 2);
    auto outLayout = Ranges::Layout(2, 2);

    GenerateWisdom<Complex, Complex>(inLayout, outLayout, Measure, true);
  }
}
