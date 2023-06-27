#include <algorithm>
#include <complex>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <random>
#include <vector>

#include "FFTW.h"

template <typename Float>
void FFTW1DTest(int);

int main() {
  int n = 512;

  std::cout << "Test for floats:       ";
  FFTW1DTest<float>(n);

  std::cout << "Test for doubles:      ";
  FFTW1DTest<double>(n);

  std::cout << "Test for long doubles: ";
  FFTW1DTest<long double>(n);
}

template <typename Float>
void FFTW1DTest(int n) {
  using Complex = std::complex<Float>;
  using Vector = std::vector<Complex>;
  
  
  // Initialise the vectors.
  Vector in(n), out(n), check(n);

  // Form the plans.
  auto flag = FFTW::PlanFlag::Measure;
  FFTW::Plan<Float> forward_plan(in.begin(), in.end(), out.begin(),
                                 FFTW::DirectionFlag::Forward, flag);

  FFTW::Plan<Float> backward_plan(out.begin(), out.end(), check.begin(),
                                  FFTW::DirectionFlag::Backward, flag);

  // Set up random number generator.
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<Float> d{0., 1.};

  // Set the input values.
  for (auto& val : in)
    val = {d(gen), d(gen)};


  // Execute the plans.
  forward_plan.execute();
  backward_plan.execute();

  // Normalise the inverse transformation.
  auto norm = static_cast<Float>(1) / static_cast<Float>(n);
  std::transform(check.cbegin(), check.cend(), check.begin(),
                 [norm](Complex x) -> Complex { return x * norm; });

  // Compute the maximum residual value.
  std::transform(in.begin(), in.end(), check.begin(), in.begin(),
                 std::minus<>());
  auto max = std::abs(*std::max_element(
      in.begin(), in.end(),
      [](Complex x, Complex y) { return std::abs(x) < std::abs(y); }));

  // Compare to 20 times the difference between 1 and the next representable Float.
  constexpr float eps = 20 * std::numeric_limits<Float>::epsilon();
  if (max < eps) {
    std::cout << "passed\n";
  } else {
    std::cout << "failed\n";
  }
};
