#ifndef FFTWPP_TEST1DR2C_GUARD_H
#define FFTWPP_TEST1DR2C_GUARD_H

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

  auto inView = FFTWpp::MakeDataView1D(in);
  auto outView = FFTWpp::MakeDataView1D(out);
  auto checkView = FFTWpp::MakeDataView1D(check);

  auto forward_plan = FFTWpp::Plan(inView, outView, flag);
  auto backward_plan = FFTWpp::Plan(outView, checkView, flag);

  // Set the input values
  MakeRealData(in.begin(), in.end());

  // Execute the plans.
  NewData ? forward_plan.Execute(inView, outView) : forward_plan.Execute();
  NewData ? backward_plan.Execute(outView, checkView) : backward_plan.Execute();

  // Normalise the transformation.
  auto norm = backward_plan.Normalisation();
  std::transform(check.begin(), check.end(), check.begin(),
                 [norm](auto x) { return norm * x; });

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

#endif  // FFTWPP_TEST1DR2C_GUARD_H
