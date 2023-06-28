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
  
  // generate a random size for the data
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> d(100, 10000); 
  int n = d(gen);

  
  // Initialise the vectors.
  RealVector in(n), check(n);
  ComplexVector out(n / 2 + 1);

  // Form the plans.
  auto flag = FFTW::PlanFlag::Measure;
  FFTW::Plan forward_plan(in.begin(), in.end(), out.begin(), flag);
  FFTW::Plan backward_plan(out.begin(), check.begin(), check.end(), flag);


  
  // Set the input values
  MakeRealData(in.begin(), in.end());

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
  auto max = std::abs(*std::max_element(in.begin(), in.end()));

  // Compare to 100 times the difference between 1 and the next representable
  // Float.
  constexpr auto eps = 100 * std::numeric_limits<Float>::epsilon();
  
  // Return 0 if passed, 1 otherwise.
  return max < eps ? 0 : 1;
}
