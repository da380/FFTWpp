#ifndef FFTWPlan_GUARD_H
#define FFTWPlan_GUARD_H

#include <variant>

#include "fftw3.h"

namespace FFTW {


  
  
  template<std::floating_point Precision>
  class Plan{

  public:


    Plan();
    

    // Constructor for 1D complex data using iterators.
        template<ComplexIterator InputIt, ComplexIterator OutputIt>
        Plan(InputIt first, InputIt last, OutputIt d_first,DirectionFlag
	     direction,PlanFlag flag);
    
    // Destructor. N
    ~Plan(){
      DestroyPlan();      
    }
      
    
  private:

    // Store the plan as a std::variant.
    std::variant<fftwf_plan,fftw_plan,fftwl_plan> plan;

    // Get plan in fftw3 form.
    auto getPlan();

    // Destroy the plan.
    void DestroyPlan();

    
  };


  template<std::floating_point Precision>
  auto Plan<Precision>::getPlan()
    {
      if constexpr(IsSingle<Precision>){
	return std::get<fftwf_plan>(plan);
      }
      if constexpr(IsDouble<Precision>){
	return std::get<fftw_plan>(plan);
      }
      if constexpr(IsQuadruple<Precision>){
	return std::get<fftwl_plan>(plan);		
      }      
    }


  template<std::floating_point Precision>
  void Plan<Precision>::DestroyPlan()
    {
      if constexpr(IsSingle<Precision>){
	fftwf_destroy_plan(getPlan());
      }
      if constexpr(IsDouble<Precision>){
	fftw_destroy_plan(getPlan());
      }
      if constexpr(IsQuadruple<Precision>){
	fftwl_destroy_plan(getPlan());
      }      
    }

  
  template<std::floating_point Precision>
  template<ComplexIterator InputIt, ComplexIterator OutputIt>
  Plan<Precision>::Plan(InputIt first, InputIt last, OutputIt d_first,DirectionFlag
	       direction,PlanFlag flag)
  {
  }

 
  
}  // namespace FFTW

#endif  // FFTWPlan1D_GUARD_H
