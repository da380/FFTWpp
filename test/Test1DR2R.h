#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "FFTW.h"
#include "MakeRealData.h"


template <typename Float>
int Test1DR2R(bool NewData = false) {
  using Vector = FFTW::vector<Float>;

  
  
  // generate a random size for the data
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<> d(100, 10000); 
  int n = d(gen);

  
  // Initialise the vectors.
  Vector in(n), check(n), out(n);


  
  // Form the plans.
  auto flag = FFTW::PlanFlag::Measure;
  auto forward = FFTW::DirectionFlag::Forward;
  auto backward = FFTW::DirectionFlag::Backward;
  //  FFTW::Plan forward_plan(n, in.begin(), out.begin(), forward, flag);
  FFTW::Plan1D forward_plan(n, in.begin(), out.begin(), forward, flag);
  FFTW::Plan1D backward_plan(n, out.begin(), check.begin(), backward, flag);


  
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
