#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

int main() {
  using Real = double;
  using Complex = std::complex<Real>;

  // Set the input and output arrays.
  auto n = 5;
  auto in = std::vector<Real>(n, 1);
  auto out = std::vector<Complex>(n / 2 + 1);

  // Raw fftw3.
  {
    // Form pointers to the data.
    auto inPointer = in.data();
    auto outPointer = reinterpret_cast<fftw_complex*>(out.data());

    // For the plans.
    auto plan = fftw_plan_dft_r2c_1d(
        n, in.data(), reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_ESTIMATE);

    // Execute the plans.
    fftw_execute(plan);
  }

  // Minimal usage of FFTWpp.
  {
    using namespace FFTWpp;
    auto plan = MakePlan(n, in.data(), out.data(), FFTW_ESTIMATE);
    Execute(plan);
  }

  // Full usage of FFTWpp.
  {
    using namespace FFTWpp::Testing;
    auto plan = Plan(View(in), View(out), Estimate);
    plan.Execute();
  }
}
