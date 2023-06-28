#include <algorithm>
#include <complex>
#include <concepts>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "FFTW.h"

template <typename Iter>
requires FFTW::ComplexIterator<Iter>
void MakeComplexData(Iter first, Iter last) {
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<typename Iter::value_type::value_type> d{0., 1.};
  for (; first != last; first++) {
    *first = {d(gen), d(gen)};
  }
}

template <std::floating_point Float>
int FFTW1DTest(bool NewData = false) {
  using Complex = std::complex<Float>;
  using Vector = FFTW::vector<Complex>;

  int n = pow(2, 14);

  // Initialise the vectors.
  Vector in(n), out(n), check(n);

  // Form the plans.
  auto flag = FFTW::PlanFlag::Measure;
  FFTW::Plan forward_plan(in.begin(), in.end(), out.begin(),
                          FFTW::DirectionFlag::Forward, flag);

  FFTW::Plan backward_plan(out.begin(), out.end(), check.begin(),
                           FFTW::DirectionFlag::Backward, flag);

  // Set the input values
  MakeComplexData(in.begin(), in.end());

  // Execute the plans.
  NewData ? forward_plan.execute(in.begin(), out.begin())
          : forward_plan.execute();
  NewData ? backward_plan.execute(out.begin(), check.begin())
          : backward_plan.execute();

  // Normalise the inverse transformation.
  backward_plan.normalise(check.begin(), check.end());

  // Compute the maximum residual value.
  std::transform(in.begin(), in.end(), check.begin(), in.begin(),
                 std::minus<>());
  auto max = std::abs(*std::max_element(
      in.begin(), in.end(),
      [](Complex x, Complex y) { return std::abs(x) < std::abs(y); }));

  // Compare to 20 times the difference between 1 and the next representable
  // Float.
  constexpr auto eps = 20 * std::numeric_limits<Float>::epsilon();

  // Return 0 if passed, 1 otherwise.
  return max < eps ? 0 : 1;
}
