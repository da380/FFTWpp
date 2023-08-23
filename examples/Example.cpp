#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

int main() {
  using Float = double;
  using Complex = std::complex<Float>;
  using RealVector = FFTWpp::vector<Float>;
  using ComplexVector = FFTWpp::vector<Complex>;

  // generate a random size for the data
  int n = 10;

  // Initialise the vectors.
  RealVector in(n), check(n);
  ComplexVector out(n / 2 + 1);

  // Form the plans.
  auto flag = FFTWpp::Measure;

  auto forward_plan = FFTWpp::MakePlan1D(in, out, flag);

  auto backward_plan = FFTWpp::MakePlan1D(out, check, flag);

  for (auto& x : in) x = 1;

  forward_plan.execute();
  backward_plan.execute();

  for (auto& x : in) std::cout << x << std::endl;
  std::cout << std::string(20, '=') << std::endl;

  for (auto& x : check) std::cout << x << std::endl;
}
