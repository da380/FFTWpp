#ifndef FFTWPP_TRANSFORM_GUARD_H
#define FFTWPP_TRANSFORM_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <numeric>
#include <variant>

#include "Concepts.h"
#include "Flags.h"
#include "Memory.h"
#include "Plan.h"
#include "Wisdom.h"
#include "fftw3.h"

namespace FFTWpp {

template <std::ranges::random_access_range InRange,
          std::ranges::random_access_range OutRange>
requires C2CIteratorPair<std::ranges::iterator_t<InRange>,
                         std::ranges::iterator_t<OutRange>> or
    C2RIteratorPair<std::ranges::iterator_t<InRange>,
                    std::ranges::iterator_t<OutRange>> or
    R2CIteratorPair<std::ranges::iterator_t<InRange>,
                    std::ranges::iterator_t<OutRange>>
void Transform1D(InRange in, OutRange out, Direction direction = Forward) {
  auto n = in.size();
  auto inView = DataView(std::begin(in), std::end(in), 1, {n}, 1, {n}, 1, 1);
  auto m = out.size();
  auto outView = DataView(std::begin(out), std::end(out), 1, {n}, 1, {n}, 1, 1);
  auto plan = Plan(inView, outView, Estimate, direction);
  plan.Execute();
}

template <std::ranges::random_access_range InRange,
          std::ranges::random_access_range OutRange>
requires R2RIteratorPair<std::ranges::iterator_t<InRange>,
                         std::ranges::iterator_t<OutRange>>
void Transform1D(InRange in, OutRange out, Kind kind,
                 Direction direction = Forward) {
  auto n = in.size();
  auto inView = DataView(std::begin(in), std::end(in), 1, {n}, 1, {n}, 1, 1);
  auto m = out.size();
  auto outView = DataView(std::begin(out), std::end(out), 1, {n}, 1, {n}, 1, 1);
  auto plan = Plan(inView, outView, Estimate, {kind}, direction);
  plan.Execute();
}

}  // namespace FFTWpp

#endif  // FFTWPP_TRANSFORM_GUARD_H
