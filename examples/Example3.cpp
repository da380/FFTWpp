
#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

/*---------------------------------------------------------//

This example illustrates differences in the API for simple
multi-dimensional transformations in the three possible cases:

(1) complex-to-complex;
(2) real-to-complex and complex-to-real;
(3) real-to-real of various kinds.

In each case the data is assumed to be stored
contiguously following the row-major convention that
is the default within fftw.

Unlike the 1D cases, when forming the views to the data,
we need to pass additional information on the layout.
However, because of the default storage assumption, this
information takes a very simple form.

For complex-to-complex and real-to-complex transformations, the
constructor for  Ranges::View need only be passed the dimensions
as additional arguments.

For real-to-real transformations, it is also necessary to pass
in the transformation kind along each direction.

As is shown in these examples, the necessary information
can be passed to the constructor in two mains ways:

(1) using variadic templated functions;
(2) using suitable ranges.

Either case can be useful, and we illustrate some helper
functions that simplify the steps.

//----------------------------------------------------------*/

int main() {
  using namespace FFTWpp;

  // Precision can be set freely between float, double and long double.
  using Real = double;
  using Complex = std::complex<Real>;

  //------------------------------------------------//
  //     2D complex-complex transformation pair     //
  //------------------------------------------------//
  {
    // Set dimensions.
    auto n0 = 20;
    auto n1 = 30;

    // Set the data sizes.
    auto [inSize, outSize] = FFTWpp::DataSize<Complex, Complex>(n0, n1);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Complex>(inSize);
    auto out = FFTWpp::vector<Complex>(outSize);
    auto copy = FFTWpp::vector<Complex>(inSize);

    // Form the plans. Here the dimensions are passed directly to the
    // constructors for the Ranges::View class.
    auto planForward = Ranges::Plan(
        Ranges::View(in, n0, n1), Ranges::View(out, n0, n1), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out, n0, n1), Ranges::View(copy, n0, n1),
                     Measure, Backward);

    // Set values for in.
    FFTWpp::RandomiseValues(in);

    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();

    // Print the error on the transform pair.
    std::cout << std::ranges::max(std::ranges::views::zip_transform(
                     [&planBackward](auto x, auto y) {
                       return std::abs(x - y * planBackward.Normalisation());
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
  }

  //------------------------------------------------//
  //       3D real-complex transformation pair      //
  //------------------------------------------------//
  {
    // Set dimension.
    auto n0 = 10;
    auto n1 = 40;
    auto n2 = 5;

    //  Get data sizes.
    auto [inSize, outSize] = FFTWpp::DataSize<Real, Complex>(n0, n1, n2);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Real>(inSize);
    auto out = FFTWpp::vector<Complex>(outSize);
    auto copy = FFTWpp::vector<Real>(inSize);

    // Get data dimensions as integer-valued ranges.
    auto [inDimensions, outDimensions] =
        FFTWpp::DataDimensions<Real, Complex>(n0, n1, n2);

    // Form the plans. Here we pass the constructor the data dimensions
    // as an integer range.
    auto planForward = Ranges::Plan(Ranges::View(in, inDimensions),
                                    Ranges::View(out, outDimensions), Measure);
    auto planBackward = Ranges::Plan(Ranges::View(out, outDimensions),
                                     Ranges::View(copy, inDimensions), Measure);

    // Set values for in.
    FFTWpp::RandomiseValues(in);

    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();

    // Print the error on the transform pair.
    std::cout << std::ranges::max(std::ranges::views::zip_transform(
                     [&planBackward](auto x, auto y) {
                       return std::abs(x - y * planBackward.Normalisation());
                     },
                     std::ranges::views::all(in),
                     std::ranges::views::all(copy)))
              << std::endl;
  }

  /*

    //------------------------------------------------//
    //         1D real-real transformation pair       //
    //------------------------------------------------//
    {
      // Set dimension.
      auto n = 200;

      //  Get data sizes.
      auto [inSize, outSize] = FFTWpp::DataSize<Real, Real>(n);

      // Allocate in, out, and copy arrays.
      auto in = FFTWpp::vector<Real>(inSize);
      auto out = FFTWpp::vector<Real>(outSize);
      auto copy = FFTWpp::vector<Real>(inSize);

      // Set the transform kind.
      auto kind = REDFT01;

      // Form the plans. Note the for the inverse transformation we
      // deduce the kind from that of the forward via the "Inverse"
      // method within the RealKind class.
      auto planForward =
          Ranges::Plan(Ranges::View(in), Ranges::View(out), kind, Measure);
      auto planBackward = Ranges::Plan(Ranges::View(out), Ranges::View(copy),
                                       kind.Inverse(), Measure);

      // Set values for in.
      FFTWpp::RandomiseValues(in);

      // Execute the plans.
      planForward.Execute();
      planBackward.Execute();

      // Print the error on the transform pair.
      std::cout << std::ranges::max(std::ranges::views::zip_transform(
                       [&planBackward](auto x, auto y) {
                         return std::abs(x - y * planBackward.Normalisation());
                       },
                       std::ranges::views::all(in),
                       std::ranges::views::all(copy)))
                << std::endl;
    }
    */
}