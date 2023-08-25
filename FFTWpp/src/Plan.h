#ifndef FFTWPP_PLAN_GUARD_H
#define FFTWPP_PLAN_GUARD_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <numeric>
#include <variant>

#include "Concepts.h"
#include "Flags.h"
#include "Memory.h"
#include "fftw3.h"

namespace FFTWpp {

template <typename InputView, typename OutputView, typename PlanFlagExpression>
class Plan {
 public:
  using InputIt = InputView::iterator;
  using OutputIt = OutputView::iterator;
  using Float = IteratorPrecision<InputIt>;
  using InputValueType = std::iter_value_t<InputIt>;
  using OutputValueType = std::iter_value_t<OutputIt>;
  using kind_value_type = decltype(FFTW_HC2R);

  // Constructors
  Plan() = delete;

  // Complex to complex constructor.
  Plan(InputView in, OutputView out, PlanFlagExpression flag,
       Direction direction) requires C2CIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{direction} {
    assert(in.Transformable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                 in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                 out.Embed(), out.Stride(), out.Dist(),
                                 direction(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                out.Embed(), out.Stride(), out.Dist(),
                                direction(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                 in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                 out.Embed(), out.Stride(), out.Dist(),
                                 direction(), flag());
    }
  }

  // Complex to real constructor.
  Plan(InputView in, OutputView out, PlanFlagExpression flag,
       Direction direction = Backward) requires
      C2RIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{Backward} {
    assert(in.Transformable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft_c2r(
          in.Rank(), out.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft_c2r(
          in.Rank(), out.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft_c2r(
          in.Rank(), out.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }
  }

  // Real to complex constructor.
  Plan(InputView in, OutputView out, PlanFlagExpression flag,
       Direction direction = Forward) requires
      R2CIteratorPair<InputIt, OutputIt>
      : in{in}, out{out}, flag{flag}, direction{Forward} {
    assert(in.Transformable(out));
    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_dft_r2c(
          in.Rank(), in.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_dft_r2c(
          in.Rank(), in.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_dft_r2c(
          in.Rank(), in.N(), in.HowMany(), in.Data(), in.Embed(), in.Stride(),
          in.Dist(), out.Data(), out.Embed(), out.Stride(), out.Dist(), flag());
    }
  }

  // Real to real constructors
  Plan(InputView in, OutputView out, PlanFlagExpression flag,
       Direction direction,
       std::vector<R2R> kinds) requires R2RIteratorPair<InputIt, OutputIt>
      : in{in},
        out{out},
        flag{flag},
        direction{direction},
        kinds{std::make_shared<std::vector<R2R>>(kinds)} {
    assert(in.Transformable(out));
    assert(kinds.size() == in.Rank());
    std::vector<kind_value_type> k(in.Rank());
    std::transform(this->kinds->begin(), this->kinds->end(), k.begin(),
                   [direction](auto kind) { return kind(direction); });

    if constexpr (IsSingle<Float>) {
      plan = fftwf_plan_many_r2r(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                 in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                 out.Embed(), out.Stride(), out.Dist(), &k[0],
                                 flag());
    }

    if constexpr (IsDouble<Float>) {
      plan = fftw_plan_many_r2r(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                out.Embed(), out.Stride(), out.Dist(), &k[0],
                                flag());
    }

    if constexpr (IsLongDouble<Float>) {
      plan = fftwl_plan_many_r2r(in.Rank(), in.N(), in.HowMany(), in.Data(),
                                 in.Embed(), in.Stride(), in.Dist(), out.Data(),
                                 out.Embed(), out.Stride(), out.Dist(), &k[0],
                                 flag());
    }
  }

  // return plan as an appropriate fftw3 pointer.
  auto operator()() const {
    if constexpr (IsSingle<Float>) {
      return std::get<fftwf_plan>(plan);
    }
    if constexpr (IsDouble<Float>) {
      return std::get<fftw_plan>(plan);
    }
    if constexpr (IsLongDouble<Float>) {
      return std::get<fftwl_plan>(plan);
    }
  }

  auto Normalisation() {
    return static_cast<OutputValueType>(1) /
           static_cast<OutputValueType>(std::reduce(
               out.NView().begin(), out.NView().end(), 1, std::multiplies<>()));
  }

  auto Normalisation() requires R2RIteratorPair<InputIt, OutputIt> {
    int dim = 1;
    auto nIt = in.NView().begin();
    auto kindIt = kinds->begin();
    while (nIt != in.NView().end()) {
      dim *= (kindIt++)->LogicalSize(*nIt++);
    }
    return static_cast<OutputValueType>(1) / static_cast<OutputValueType>(dim);
  }

  // Execute the plan.
  void execute() {
    if constexpr (IsSingle<Float>) {
      fftwf_execute(this->operator()());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute(this->operator()());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute(this->operator()());
    }
  }

  // Execute the plan given new complex-complex data.
  void execute(InputView newIn,
               OutputView newOut) requires C2CIteratorPair<InputIt, OutputIt> {
    assert(in.EqualStorage(newIn));
    assert(out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new complex-real data.
  void execute(InputView newIn,
               OutputView newOut) requires C2RIteratorPair<InputIt, OutputIt> {
    assert(in.EqualStorage(newIn));
    assert(out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new real-complex data.
  void execute(InputView newIn,
               OutputView newOut) requires R2CIteratorPair<InputIt, OutputIt> {
    assert(in.EqualStorage(newIn));
    assert(out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new real-real data.
  void execute(InputView newIn,
               OutputView newOut) requires R2RIteratorPair<InputIt, OutputIt> {
    assert(in.EqualStorage(newIn));
    assert(out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Float>) {
      fftwf_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Float>) {
      fftw_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Destructor.
  ~Plan() {
    if constexpr (IsSingle<Float>) {
      fftwf_destroy_plan(this->operator()());
    }
    if constexpr (IsDouble<Float>) {
      fftw_destroy_plan(this->operator()());
    }
    if constexpr (IsLongDouble<Float>) {
      fftwl_destroy_plan(this->operator()());
    }
  }

 private:
  // Store data views.
  InputView in;
  OutputView out;

  // Store transform options
  Direction direction;
  PlanFlagExpression flag;
  std::shared_ptr<std::vector<R2R>> kinds;

  // Store the plan as a std::variant.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> plan;
};

// Returns a plan for a 1D transformation given data in range format.
template <typename InputRange, typename OutputRange,
          typename PlanFlagExpression>
auto MakePlan1D(InputRange& in, OutputRange& out, PlanFlagExpression flag,
                Direction direction = Forward) {
  return Plan(MakeDataView1D(in), MakeDataView1D(out), flag, direction);
}

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
