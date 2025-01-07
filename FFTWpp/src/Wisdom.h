#ifndef FFTWPP_WISDOM_GUARD_H
#define FFTWPP_WISDOM_GUARD_H

#include <cassert>
#include <string>

#include "NumericConcepts/Numeric.hpp"
#include "NumericConcepts/Ranges.hpp"
#include "Options.h"
#include "Plan.h"
#include "Views.h"
#include "fftw3.h"

namespace FFTWpp {

void ExportWisdom(const std::string& filename) {
  int io = fftw_export_wisdom_to_filename(filename.c_str());
  assert(io == 0);
}

void ImportWisdom(const std::string& filename) {
  int io = fftw_import_wisdom_from_filename(filename.c_str());
  assert(io == 0);
}

void ForgetWisdom() { fftw_forget_wisdom(); }

template <NumericConcepts::RealOrComplex InType,
          NumericConcepts::RealOrComplex OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  if constexpr (NumericConcepts::Complex<InType> &&
                NumericConcepts::Complex<OutType>) {
    auto plan = Ranges::Plan(inView, outView, flag, Forward);
    plan = Ranges::Plan(outView, inView, flag, Backward);
  }
  if constexpr ((NumericConcepts::Real<InType> &&
                 NumericConcepts::Complex<OutType>) ||
                (NumericConcepts::Complex<InType> &&
                 NumericConcepts::Real<OutType>)) {
    auto planForward = Ranges::Plan(inView, outView, flag);
    auto planBackward = Ranges::Plan(outView, inView, flag);
  }
}

template <NumericConcepts::Real InType, NumericConcepts::Real OutType>
requires NumericConcepts::SamePrecision<InType, OutType>
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    std::initializer_list<RealKind> kinds, Flag flag) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);
  auto planForward = Ranges::Plan(inView, outView, kinds, flag);
  auto kindsBackward = kinds;
  std::ranges::views::all(kindsBackward) |
      std::ranges::views::transform([](auto kind) { return kind.Inverse(); });
  auto planBackward = Ranges::Plan(outView, inView, kindsBackward, flag);
}

}  // namespace FFTWpp

#endif  // FFTWPP_WISDOM_GUARD_H
