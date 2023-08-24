#ifndef FFTWPP_TEST1DC2C_GUARD_H
#define FFTWPP_TEST1DC2C_GUARD_H

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
int Test1DC2C(bool NewData = false) {
  using Complex = std::complex<Float>;
  using Vector = FFTWpp::vector<Complex>;

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

  auto forward_plan = FFTWpp::Plan(inView, outView, flag, FFTWpp::Forward);

  auto backward_plan = FFTWpp::Plan(outView, checkView, flag, FFTWpp::Backward);

  // Set the input values
  MakeComplexData(in.begin(), in.end());

  NewData ? forward_plan.execute(inView, outView) : forward_plan.execute();
  NewData ? backward_plan.execute(outView, checkView, FFTWpp::Normalised)
          : backward_plan.execute(FFTWpp::Normalised);

  // Compute the maximum residual value.
  std::transform(in.begin(), in.end(), check.begin(), in.begin(),
                 std::minus<>());
  auto max = std::abs(*std::max_element(
      in.begin(), in.end(),
      [](Complex x, Complex y) { return std::abs(x) < std::abs(y); }));

  // Compare to 100 times the difference between 1 and the next representable
  // Float.
  constexpr auto eps = 100 * std::numeric_limits<Float>::epsilon();

  // Return 0 if passed, 1 otherwise.
  std::cout << max / eps << std::endl;
  return max < eps ? 0 : 1;
}

#endif  // FFTWPP_TEST1DC2C_GUARD_H
