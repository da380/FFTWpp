#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "MakeData.h"

template <typename Float>
int Test1DR2C(bool NewData = false) {
  using Complex = std::complex<Float>;
  using RealVector = FFTWpp::vector<Float>;
  using ComplexVector = FFTWpp::vector<Complex>;

  // generate a random size for the data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(100, 1000);
  int n = d(gen);

  // Initialise the vectors.
  RealVector in(n), check(n);
  ComplexVector out(n / 2 + 1);

  // Form the plans.
  auto flag = FFTWpp::Measure;

  auto inRef = FFTWpp::MakeDataReference1D(in.begin(), in.end());
  auto outRef = FFTWpp::MakeDataReference1D(out.begin(), out.end());
  auto checkRef = FFTWpp::MakeDataReference1D(check.begin(), check.end());

  auto forward_plan = FFTWpp::Plan(inRef, outRef, flag);

  auto backward_plan = FFTWpp::Plan(outRef, checkRef, flag);

  // Set the input values
  MakeRealData(in.begin(), in.end());

  // Execute the plans.
  NewData ? forward_plan.execute(inRef, outRef) : forward_plan.execute();
  NewData ? backward_plan.execute(outRef, checkRef) : backward_plan.execute();

  // Normalise the inverse transformation.
  checkRef.normalise();

  // Compute the maximum residual value.
  std::transform(in.begin(), in.end(), check.begin(), in.begin(),
                 std::minus<>());
  auto max = std::abs(*std::max_element(in.begin(), in.end()));

  // Compare to 100 times the difference between 1 and the next representable
  // Float.
  constexpr auto eps = 100 * std::numeric_limits<Float>::epsilon();

  // Return 0 if passed, 1 otherwise.
  return max < eps ? 0 : 1;

  return 0;
}
