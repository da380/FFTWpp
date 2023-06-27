#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "FFTW.h"

template <typename Iter>
requires FFTW::RealIterator<Iter>
void MakeRealData(Iter first, Iter last) {
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::normal_distribution<typename Iter::value_type> d{0., 1.};
  for (; first != last; first++) {
    *first = d(gen);
  }
}

template <typename Float>
int FFTW1DRealTest(bool NewData = false) {
  using Complex = std::complex<Float>;
  using RealVector = FFTW::vector<Float>;
  using ComplexVector = FFTW::vector<Complex>;

  int n = 512;

  // Initialise the vectors.
  RealVector in(n), check(n);
  ComplexVector out(n / 2 + 1);

  // Form the plans.
  auto flag = FFTW::PlanFlag::Measure;
  FFTW::Plan<Float> forward_plan(in.begin(), in.end(), out.begin(), flag);
  FFTW::Plan<Float> backward_plan(out.begin(), out.end(), check.begin(), flag);

  // Set the input values
  MakeRealData(in.begin(), in.end());

  // Execute the plans.
  NewData ?    forward_plan.execute(in.begin(),out.begin()) :   forward_plan.execute();
  NewData ? backward_plan.execute(out.begin(),check.begin()) : backward_plan.execute();


  // Normalise the inverse transformation.
  auto norm = static_cast<Float>(1) / static_cast<Float>(n);
  std::transform(check.cbegin(), check.cend(), check.begin(),
                 [norm](Float x) { return x * norm; });

  // Compute the maximum residual value.
  std::transform(in.begin(), in.end(), check.begin(), in.begin(),
                 std::minus<>());
  auto max = std::abs(*std::max_element(in.begin(), in.end()));

  // Compare to 20 times the difference between 1 and the next representable
  // Float.
  constexpr auto eps = 20 * std::numeric_limits<Float>::epsilon();

  // Return 0 if passed, 1 otherwise.
  return max < eps ? 0 : 1;
}
