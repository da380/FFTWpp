#ifndef FFTWPP_TEST1D_GUARD_H
#define FFTWPP_TEST1D_GUARD_H

#include <FFTWpp/Ranges>
#include <complex>
#include <random>
#include <ranges>

template <FFTWpp::IsScalar InType, FFTWpp::IsScalar OutType>
auto Test1D(FFTWpp::RealKind kind = FFTWpp::R2HC) {
  using namespace FFTWpp;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(10, 1000);
  int n = d(gen);
  auto [inSize, outSize] = DataSize<InType, OutType>(n);
  auto in = vector<InType>(inSize);
  auto out = vector<OutType>(outSize);
  auto copy = in;
  if constexpr (IsComplex<InType> && IsComplex<OutType>) {
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure, Backward);
    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();
    return CheckValues(in, copy, planBackward.Normalisation());
  } else if constexpr (IsReal<InType> && IsComplex<OutType>) {
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure);
    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();
    return CheckValues(in, copy, planBackward.Normalisation());
  } else if constexpr (IsReal<InType> && IsReal<OutType>) {
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure, kind);
    auto planBackward = Ranges::Plan(Ranges::View(out), Ranges::View(copy),
                                     Measure, kind.Inverse());
    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();
    return CheckValues(in, copy, planBackward.Normalisation());
  }
}

#endif
