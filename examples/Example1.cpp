
#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

/*---------------------------------------------------------//

This example compares the use of FFTWpp to direct calls
of fftw3 functions  using both the minimal interface from
<FTTWpp/Core> and that within <FFTpp/Ranges>. Note that
inclusion of the latter header implies that of the former.

1D real-to-complex and complex-to-real transforms are
considered. Each code block performs analogous tasks, and
prints the maximum error after forward and inverse transforming
the randomised inputs.

The direct fftw3 calls are for double precision functions.
Those from FFTWpp are to templated functions that allow
for single and long double with no change in the API.

<FTTWpp/Core> provides templated overloads of the common fftw3
functions, and allows for direct use of std::complex types. The
syntax mirrors exactly that of the underlying functions,
but with the function names uniformitised. Plans are still
of pointer-type, and it is necesarry to explicitly free
associated memory.

With <FFTWpp/Ranges> we work with a plan class that automates
memory management. In this case, plans access their
data via a view that contains information on the data
layout following the fftw3 "advanced interface". Further
details on this approach can be seen in later examples.
This approach is also fully type-safe and provides checks
on the dimensions of input arrays via assert statements.

//----------------------------------------------------------*/

int main() {
  using namespace FFTWpp;

  // Set the data types.
  using Real = double;
  using InType = Real;
  using OutType = std::complex<Real>;

  // Generate allocate input and output arrays. Note that FFTWpp::vector
  // is a version of std::vector with a custom allocator based on
  // fftw_malloc. It is not necessary to use such vectors, but doing
  // so insures the correct memory alignment for optimal transformations.
  auto n = 64;
  auto [inSize, outSize] = FFTWpp::DataSize<InType, OutType>(n);
  auto in = FFTWpp::vector<InType>(inSize);
  auto out = FFTWpp::vector<OutType>(outSize);
  auto copy = FFTWpp::vector<InType>(inSize);

  // Direct fftw3 section. Note that is Real != double this
  // code block will not work.
  {
    // Make the plans.
    auto planForward = fftw_plan_dft_r2c_1d(
        n, in.data(), reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_MEASURE);
    auto planBackward =
        fftw_plan_dft_c2r_1d(n, reinterpret_cast<fftw_complex*>(out.data()),
                             copy.data(), FFTW_MEASURE);
    // Set in values.
    FFTWpp::RandomiseValues(in);
    // Execute plans.
    fftw_execute(planForward);
    fftw_execute(planBackward);
    // Print the error on the transform pair.
    std::cout << "Raw fftw3 section:\n Error = "
              << std::ranges::max(std::ranges::views::zip_transform(
                     [n](auto x, auto y) {
                       return std::abs(x - y / static_cast<double>(n));
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
    // Delete the plans to free memory.
    fftw_destroy_plan(planForward);
    fftw_destroy_plan(planBackward);
  }

  // Transforms done using FFTWpp/Core.
  {
    // Make the plans.
    auto planForward = Plan(n, in.data(), out.data(), FFTW_MEASURE);
    auto planBackward = Plan(n, out.data(), copy.data(), FFTW_MEASURE);
    // Set in values.
    FFTWpp::RandomiseValues(in);
    // Execute the plans.
    Execute(planForward);
    Execute(planBackward);
    // Print the error on the transform pair.
    std::cout << "Minimal FFTWpp section:\n Error = "
              << std::ranges::max(std::ranges::views::zip_transform(
                     [n](auto x, auto y) {
                       return std::abs(x - y / static_cast<double>(n));
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
    // Delete the plans to free memory.
    Destroy(planForward);
    Destroy(planBackward);
  }

  // Full usage of FFTWpp.
  {
    // Make the plans.
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure);
    // Set in values.
    FFTWpp::RandomiseValues(in);
    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();
    // Print the error on the transform pair.
    std::cout << "Full FFTWpp section:\n Error = "
              << std::ranges::max(std::ranges::views::zip_transform(
                     [&planBackward](auto x, auto y) {
                       return std::abs(x - y * planBackward.Normalisation());
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
  }

  // Optionally clean up "still reachably" memory.
  FFTWpp::CleanUp();
}
