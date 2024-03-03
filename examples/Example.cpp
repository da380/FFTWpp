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

  // Raw fftw3. Note that if Real is changed from double, then
  // various terms in this block need to be modified.
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

  // Minimal usage of FFTWpp. If real is changed from double, nothing needs
  // be done in this case.
  {
    auto plan = FFTWpp::MakePlan(n, in.data(), out.data(), FFTW_ESTIMATE);
    FFTWpp::Execute(plan);
  }

  auto layout = FFTWpp::Testing::Layout(10, 20);

  std::cout << layout.Rank() << std::endl;
  for (auto val : layout.N()) std::cout << val << " ";
  std::cout << std::endl;
}
