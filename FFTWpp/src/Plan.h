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

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//                        Definition of the Plan class                       //
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename InputView, typename OutputView>
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
  Plan(InputView in, OutputView out, PlanFlag flag,
       Direction direction) requires C2CIteratorPair<InputIt, OutputIt>
      : _in{in}, _out{out}, _flag{flag}, _direction{direction} {
    assert(in.Transformable(_out));
    if constexpr (IsSingle<Float>) {
      _plan = fftwf_plan_many_dft(
          _in.Rank(), _in.N(), _in.HowMany(), _in.Data(), _in.Embed(),
          _in.Stride(), _in.Dist(), _out.Data(), _out.Embed(), _out.Stride(),
          _out.Dist(), _direction(), _flag());
    }

    if constexpr (IsDouble<Float>) {
      _plan = fftw_plan_many_dft(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), _direction(), _flag());
    }

    if constexpr (IsLongDouble<Float>) {
      _plan = fftwl_plan_many_dft(
          _in.Rank(), _in.N(), _in.HowMany(), _in.Data(), _in.Embed(),
          _in.Stride(), _in.Dist(), _out.Data(), _out.Embed(), _out.Stride(),
          _out.Dist(), _direction(), _flag());
    }
  }

  // Complex to real constructor.
  Plan(InputView in, OutputView out, PlanFlag flag,
       Direction direction = Backward) requires
      C2RIteratorPair<InputIt, OutputIt>
      : _in{in}, _out{out}, _flag{flag}, _direction{Backward} {
    assert(_in.Transformable(_out));
    if constexpr (IsSingle<Float>) {
      _plan = fftwf_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                      _in.Data(), _in.Embed(), _in.Stride(),
                                      _in.Dist(), _out.Data(), _out.Embed(),
                                      _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsDouble<Float>) {
      _plan = fftw_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsLongDouble<Float>) {
      _plan = fftwl_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                      _in.Data(), _in.Embed(), _in.Stride(),
                                      _in.Dist(), _out.Data(), _out.Embed(),
                                      _out.Stride(), _out.Dist(), _flag());
    }
  }

  // Real to complex constructor.
  Plan(InputView in, OutputView out, PlanFlag flag,
       Direction direction = Forward) requires
      R2CIteratorPair<InputIt, OutputIt>
      : _in{in}, _out{out}, _flag{flag}, _direction{Forward} {
    assert(_in.Transformable(_out));
    if constexpr (IsSingle<Float>) {
      _plan = fftwf_plan_many_dft_r2c(_in.Rank(), _in.N(), _in.HowMany(),
                                      _in.Data(), _in.Embed(), _in.Stride(),
                                      _in.Dist(), _out.Data(), _out.Embed(),
                                      _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsDouble<Float>) {
      _plan = fftw_plan_many_dft_r2c(_in.Rank(), _in.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsLongDouble<Float>) {
      _plan = fftwl_plan_many_dft_r2c(in.Rank(), in.N(), in.HowMany(),
                                      in.Data(), in.Embed(), in.Stride(),
                                      in.Dist(), _out.Data(), _out.Embed(),
                                      _out.Stride(), _out.Dist(), _flag());
    }
  }

  // Real to real constructors
  Plan(InputView in, OutputView out, PlanFlag flag, std::vector<Kind> kinds,
       Direction direction) requires R2RIteratorPair<InputIt, OutputIt>
      : _in{in},
        _out{out},
        _flag{flag},
        _direction{direction},
        _kinds{std::make_shared<std::vector<Kind>>(kinds)} {
    assert(_in.Transformable(_out));
    assert(_kinds->size() == _in.Rank());
    std::vector<kind_value_type> k(_in.Rank());
    std::transform(this->_kinds->begin(), this->_kinds->end(), k.begin(),
                   [this](auto kind) { return kind(_direction); });

    if constexpr (IsSingle<Float>) {
      _plan = fftwf_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(),
                                  _in.Data(), _in.Embed(), _in.Stride(),
                                  _in.Dist(), _out.Data(), _out.Embed(),
                                  _out.Stride(), _out.Dist(), &k[0], _flag());
    }

    if constexpr (IsDouble<Float>) {
      _plan = fftw_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), &k[0], _flag());
    }

    if constexpr (IsLongDouble<Float>) {
      _plan = fftwl_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(),
                                  _in.Data(), _in.Embed(), _in.Stride(),
                                  _in.Dist(), _out.Data(), _out.Embed(),
                                  _out.Stride(), _out.Dist(), &k[0], _flag());
    }
  }

  // return plan as an appropriate fftw3 pointer.
  auto operator()() const {
    if constexpr (IsSingle<Float>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (IsDouble<Float>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (IsLongDouble<Float>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  auto Normalisation() {
    return static_cast<OutputValueType>(1) /
           static_cast<OutputValueType>(std::reduce(_out.NView().begin(),
                                                    _out.NView().end(), 1,
                                                    std::multiplies<>()));
  }

  auto Normalisation() requires R2RIteratorPair<InputIt, OutputIt> {
    int dim = 1;
    auto nIt = _in.NView().begin();
    auto kindIt = _kinds->begin();
    while (nIt != _in.NView().end()) {
      dim *= (kindIt++)->LogicalSize(*nIt++);
    }
    return static_cast<OutputValueType>(1) / static_cast<OutputValueType>(dim);
  }

  // Execute the plan.
  void Execute() {
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
  void Execute(InputView newIn,
               OutputView newOut) requires C2CIteratorPair<InputIt, OutputIt> {
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
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
  void Execute(InputView newIn,
               OutputView newOut) requires C2RIteratorPair<InputIt, OutputIt> {
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
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
  void Execute(InputView newIn,
               OutputView newOut) requires R2CIteratorPair<InputIt, OutputIt> {
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
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
  void Execute(InputView newIn,
               OutputView newOut) requires R2RIteratorPair<InputIt, OutputIt> {
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
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
  InputView _in;
  OutputView _out;

  // Store transform options
  Direction _direction;
  PlanFlag _flag;
  std::shared_ptr<std::vector<Kind>> _kinds;

  // Store the plan as a std::variant.
  std::variant<fftwf_plan, fftw_plan, fftwl_plan> _plan;
};

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
