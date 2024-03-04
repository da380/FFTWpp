#ifndef FFTWPP_WISDOM_GUARD_H
#define FFTWPP_WISDOM_GUARD_H

#include <cassert>
#include <string>

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

template <typename InType, typename OutType>
requires requires() {
  requires IsReal<InType> or IsComplex<InType>;
  requires IsReal<OutType> or IsComplex<OutType>;
  requires std::same_as<RemoveComplex<InType>, RemoveComplex<OutType>>;
}
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    Flag flag, bool forward, bool backward = false) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);

  if constexpr (IsComplex<InType> && IsComplex<OutType>) {
    if (forward) auto plan = Ranges::Plan(inView, outView, flag, Forward);
    if (backward) auto plan = Ranges::Plan(outView, inView, flag, Backward);
  }

  if constexpr ((IsReal<InType> && IsComplex<OutType>) ||
                (IsComplex<InType> && IsReal<OutType>)) {
    if (forward) auto plan = Ranges::Plan(inView, outView, flag);
    if (backward) auto plan = Ranges::Plan(outView, inView, flag);
  }
}

template <typename InType, typename OutType>
requires requires() {
  requires IsReal<InType> and IsReal<OutType>;
  requires std::same_as<InType, OutType>;
}
void GenerateWisdom(Ranges::Layout inLayout, Ranges::Layout outLayout,
                    std::initializer_list<RealKind> kinds, Flag flag,
                    bool forward, bool backward = false) {
  if (flag == Estimate) return;
  auto in = vector<InType>(inLayout.size());
  auto inView = Ranges::View(in, inLayout);
  auto out = vector<OutType>(outLayout.size());
  auto outView = Ranges::View(out, outLayout);

  if (forward) auto plan = Ranges::Plan(inView, outView, kinds, flag);
  if (backward) {
    auto kindsBackward = kinds;
    std::ranges::views::all(kindsBackward) |
        std::ranges::views::transform([](auto kind) { return kind.Inverse(); });
    auto plan = Ranges::Plan(outView, inView, kindsBackward, flag);
  }
}

}  // namespace FFTWpp

#endif  // FFTWPP_WISDOM_GUARD_H
