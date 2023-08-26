#ifndef FFTWPP_TEST1DR2R_GUARD_H
#define FFTWPP_TEST1DR2R_GUARD_H

#include <FFTWpp/All>
#include <algorithm>
#include <complex>
#include <concepts>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "MakeData.h"

template <std::floating_point Float, bool Ranges = false>
int Test1DR2R(bool NewData = false) {
  using Vector = FFTWpp::vector<Float>;

  // generate a random size for the data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(10, 1000);
  int n = d(gen);

  // Initialise the vectors.
  Vector in(n), out(n), check(n);

  // Form the plans.
  auto flag = FFTWpp::Measure;

  auto view = FFTWpp::MakeDataView1D(in);
  auto inView = view;

  auto outView = FFTWpp::MakeDataView1D(out);
  auto checkView = FFTWpp::MakeDataView1D(check);

  auto kinds = std::vector<FFTWpp::R2R>(1, FFTWpp::DCTI);

  auto forward_plan =
      FFTWpp::Plan(inView, outView, flag, FFTWpp::Forward, kinds);

  auto backward_plan =
      FFTWpp::Plan(outView, checkView, flag, FFTWpp::Backward, kinds);

  // Set the input values
  MakeRealData(in.begin(), in.end());

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
  std::cout << max / eps << std::endl;
  return max < eps ? 0 : 1;
}

#endif  // FFTWPP_TEST1DR2R_GUARD_H
