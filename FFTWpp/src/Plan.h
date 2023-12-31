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
#include "Wisdom.h"
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
  using Precision = IteratorPrecision<InputIt>;
  using input_value_type = std::iter_value_t<InputIt>;
  using output_value_type = std::iter_value_t<OutputIt>;
  using kind_value_type = decltype(FFTW_HC2R);

  // Remove default constructor;
  Plan() = delete;

  // Constructor for C2C, C2R, R2C.
  Plan(InputView in, OutputView out, PlanFlag flag,
       Direction direction = Forward) requires
      C2CIteratorPair<InputIt, OutputIt> or
      C2RIteratorPair<InputIt, OutputIt> or R2CIteratorPair<InputIt, OutputIt>
      : _in{in},
        _out{out},
        _flag{flag},
        _direction{direction},
        _plan{MakePlan()} {
    assert(!IsNull());
  }

  // Constructor for R2R.
  Plan(InputView in, OutputView out, PlanFlag flag, std::vector<Kind> kinds,
       Direction direction = Forward) requires
      R2RIteratorPair<InputIt, OutputIt>
      : _in{in},
        _out{out},
        _flag{flag},
        _direction{direction},
        _kinds{std::make_shared<std::vector<Kind>>(kinds)},
        _plan{MakePlan()} {}

  // Copy constructor.
  Plan(Plan const& other)
      : _in{other._in},
        _out{other._out},
        _flag{other._flag},
        _direction{other._direction},
        _kinds{other._kinds},
        _plan{MakePlan()} {}

  // Move constructor.
  Plan(Plan&& other)
      : _in{std::move(other._in)},
        _out{std::move(other._out)},
        _flag{std::move(other._flag)},
        _direction{std::move(other._direction)},
        _kinds{std::move(other._kinds)},
        _plan{MakePlan()} {
    other.Destroy();
  }

  // Copy assignment.
  Plan& operator=(Plan const& other) {
    Destroy();
    _in = other._in;
    _out = other._out;
    _flag = other._flag;
    _direction = other._direction;
    _kinds = other._kinds;
    _plan = MakePlan();
    return *this;
  }

  // Move assignment.
  Plan& operator=(Plan&& other) {
    Destroy();
    other.Destroy();
    _in = std::move(other._in);
    _out = std::move(other._out);
    _flag = std::move(other._flag);
    _direction = std::move(other._direction);
    _kinds = std::move(other._kinds);
    _plan = MakePlan();
    return *this;
  }

  // Destructor.
  ~Plan() { Destroy(); }

  // return plan as an appropriate fftw3 pointer.
  auto operator()() const {
    if constexpr (IsSingle<Precision>) {
      return std::get<fftwf_plan>(_plan);
    }
    if constexpr (IsDouble<Precision>) {
      return std::get<fftw_plan>(_plan);
    }
    if constexpr (IsLongDouble<Precision>) {
      return std::get<fftwl_plan>(_plan);
    }
  }

  // Returns true is plan is not set up.
  auto IsNull() { return operator()() == nullptr; }

  auto Normalisation() {
    return static_cast<output_value_type>(1) /
           static_cast<output_value_type>(std::reduce(_out.NView().begin(),
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
    return static_cast<output_value_type>(1) /
           static_cast<output_value_type>(dim);
  }

  // Execute the plan.
  void Execute() {
    assert(!IsNull());
    if constexpr (IsSingle<Precision>) {
      fftwf_execute(this->operator()());
    }
    if constexpr (IsDouble<Precision>) {
      fftw_execute(this->operator()());
    }
    if constexpr (IsLongDouble<Precision>) {
      fftwl_execute(this->operator()());
    }
  }

  // Execute the plan given new complex-complex data.
  void Execute(InputView newIn,
               OutputView newOut) requires C2CIteratorPair<InputIt, OutputIt> {
    assert(!IsNull());
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Precision>) {
      fftwf_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Precision>) {
      fftw_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Precision>) {
      fftwl_execute_dft(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new complex-real data.
  void Execute(InputView newIn,
               OutputView newOut) requires C2RIteratorPair<InputIt, OutputIt> {
    assert(!IsNull());
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Precision>) {
      fftwf_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Precision>) {
      fftw_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Precision>) {
      fftwl_execute_dft_c2r(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new real-complex data.
  void Execute(InputView newIn,
               OutputView newOut) requires R2CIteratorPair<InputIt, OutputIt> {
    assert(!IsNull());
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Precision>) {
      fftwf_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Precision>) {
      fftw_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Precision>) {
      fftwl_execute_dft_r2c(this->operator()(), newIn.Data(), newOut.Data());
    }
  }

  // Execute the plan given new real-real data.
  void Execute(InputView newIn,
               OutputView newOut) requires R2RIteratorPair<InputIt, OutputIt> {
    assert(!IsNull());
    assert(_in.EqualStorage(newIn));
    assert(_out.EqualStorage(newOut));
    assert(newIn.Transformable(newOut));
    if constexpr (IsSingle<Precision>) {
      fftwf_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsDouble<Precision>) {
      fftw_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
    }
    if constexpr (IsLongDouble<Precision>) {
      fftwl_execute_r2r(this->operator()(), newIn.Data(), newOut.Data());
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

  // Destroy the stored plan.
  void Destroy() {
    if constexpr (IsSingle<Precision>) {
      auto& plan = std::get<fftwf_plan>(_plan);
      if (plan != nullptr) {
        fftwf_destroy_plan(plan);
        plan = nullptr;
      }
    }
    if constexpr (IsDouble<Precision>) {
      auto& plan = std::get<fftw_plan>(_plan);
      if (plan != nullptr) {
        fftw_destroy_plan(plan);
        plan = nullptr;
      }
    }
    if constexpr (IsLongDouble<Precision>) {
      auto& plan = std::get<fftwl_plan>(_plan);
      if (plan != nullptr) {
        fftwl_destroy_plan(plan);
        plan = nullptr;
      }
    }
  }

  // Make a plan for C2C transformation.
  auto MakePlan() requires C2CIteratorPair<InputIt, OutputIt> {
    assert(_in.Transformable(_out));
    if constexpr (IsSingle<Precision>) {
      return fftwf_plan_many_dft(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), _direction(), _flag());
    }

    if constexpr (IsDouble<Precision>) {
      return fftw_plan_many_dft(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                _in.Embed(), _in.Stride(), _in.Dist(),
                                _out.Data(), _out.Embed(), _out.Stride(),
                                _out.Dist(), _direction(), _flag());
    }

    if constexpr (IsLongDouble<Precision>) {
      return fftwl_plan_many_dft(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), _direction(), _flag());
    }
  }

  // Make plan for C2R transformation
  auto MakePlan() requires C2RIteratorPair<InputIt, OutputIt> {
    assert(_in.Transformable(_out));
    if constexpr (IsSingle<Precision>) {
      return fftwf_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsDouble<Precision>) {
      return fftw_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                    _in.Data(), _in.Embed(), _in.Stride(),
                                    _in.Dist(), _out.Data(), _out.Embed(),
                                    _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsLongDouble<Precision>) {
      return fftwl_plan_many_dft_c2r(_in.Rank(), _out.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }
  }

  // Make plan for R2C transform.
  auto MakePlan() requires R2CIteratorPair<InputIt, OutputIt> {
    assert(_in.Transformable(_out));
    if constexpr (IsSingle<Precision>) {
      return fftwf_plan_many_dft_r2c(_in.Rank(), _in.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsDouble<Precision>) {
      return fftw_plan_many_dft_r2c(_in.Rank(), _in.N(), _in.HowMany(),
                                    _in.Data(), _in.Embed(), _in.Stride(),
                                    _in.Dist(), _out.Data(), _out.Embed(),
                                    _out.Stride(), _out.Dist(), _flag());
    }

    if constexpr (IsLongDouble<Precision>) {
      return fftwl_plan_many_dft_r2c(_in.Rank(), _in.N(), _in.HowMany(),
                                     _in.Data(), _in.Embed(), _in.Stride(),
                                     _in.Dist(), _out.Data(), _out.Embed(),
                                     _out.Stride(), _out.Dist(), _flag());
    }
  }

  // Make plan for R2R transform
  auto MakePlan() requires R2RIteratorPair<InputIt, OutputIt> {
    assert(_in.Transformable(_out));
    assert(_kinds->size() == _in.Rank());
    std::vector<kind_value_type> k(_in.Rank());
    std::transform(this->_kinds->begin(), this->_kinds->end(), k.begin(),
                   [this](auto kind) { return kind(_direction); });

    if constexpr (IsSingle<Precision>) {
      return fftwf_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), &k[0], _flag());
    }

    if constexpr (IsDouble<Precision>) {
      return fftw_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                _in.Embed(), _in.Stride(), _in.Dist(),
                                _out.Data(), _out.Embed(), _out.Stride(),
                                _out.Dist(), &k[0], _flag());
    }

    if constexpr (IsLongDouble<Precision>) {
      return fftwl_plan_many_r2r(_in.Rank(), _in.N(), _in.HowMany(), _in.Data(),
                                 _in.Embed(), _in.Stride(), _in.Dist(),
                                 _out.Data(), _out.Embed(), _out.Stride(),
                                 _out.Dist(), &k[0], _flag());
    }
  }
};

// Creates a dummy plan to generate wisdom for C2C, C2R, and R2C.
template <typename in_value_type, typename out_value_type, bool both = false>
void GenerateWisdom(DataLayout inLayout, DataLayout outLayout, PlanFlag flag,
                    Direction direction = Forward) {
  auto in = inLayout.FakeData<in_value_type>();
  auto out = outLayout.FakeData<out_value_type>();
  auto inView = DataView(in->begin(), in->end(), inLayout);
  auto outView = DataView(out->begin(), out->end(), outLayout);
  auto plan = Plan(inView, outView, flag, direction);
  if (both) {
    auto plan = Plan(outView, inView, flag, direction.Reverse());
  }
  return;
}

// Creates a dummy plan to generate wisdom for R2R.
template <typename in_value_type, typename out_value_type, bool both = false>
requires IsScalar<in_value_type> and IsScalar<out_value_type>
void GenerateWisdom(DataLayout inLayout, DataLayout outLayout, PlanFlag flag,
                    std::vector<Kind> kinds, Direction direction = Forward) {
  auto in = inLayout.FakeData<in_value_type>();
  auto out = outLayout.FakeData<out_value_type>();
  auto inView = DataView(in->begin(), in->end(), inLayout);
  auto outView = DataView(out->begin(), out->end(), outLayout);
  auto plan = Plan(inView, outView, flag, kinds, direction);
  if (both) {
    auto plan = Plan(outView, inView, flag, direction.Reverse());
  }
  return;
}

}  // namespace FFTWpp

#endif  // FFTWPP_PLAN_GUARD_H
