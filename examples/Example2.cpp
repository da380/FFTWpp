#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

/*---------------------------------------------------------//

This example illustrates differences in the API for 1D
transformations in the three possible cases:

(1) complex-to-complex;
(2) real-to-complex and complex-to-real;
(3) real-to-real of various kinds.

From a user perspective, the main difference is that the
constructor of the plan class takes additional arguments
in cases (1) and (3).

For complex-to-complex transformations we need to specify
the direction of the transformation. This is done using
one of two pre-defined constants within the FFTWpp namespace.
These constants are instances of a trivial class Direction
that wraps the corresponding fftw integer constants in a
type-safe manner.

For real-to-real transformations we need to specify the
transformation kind. Again, this is done from a choice of
constants within the FFTWpp namespace, this time of type
RealKind. Each instance of this class provides access to
the corresponding fftw_r2r_kind value. It also has methods
to allow the kind of the inverse transformation to be deduced,
and to determine the "logical dimension" needed to normalise
inverse transformations.

//----------------------------------------------------------*/

int main() {
  using namespace FFTWpp;

  // Precision can be set freely between float, double and long double.
  using Real = double;
  using Complex = std::complex<Real>;

  //------------------------------------------------//
  //     1D complex-complex transformation pair     //
  //------------------------------------------------//
  {
    // Set dimension.
    auto n = 200;

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Complex>(n);
    auto out = FFTWpp::vector<Complex>(n);
    auto copy = FFTWpp::vector<Complex>(n);

    // Form the plans
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure, Backward);

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
  //       1D real-complex transformation pair      //
  //------------------------------------------------//
  {
    // Set dimension.
    auto n = 200;

    //  Get data sizes.
    auto [inSize, outSize] = FFTWpp::DataSize<Real, Complex>(n);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Real>(inSize);
    auto out = FFTWpp::vector<Complex>(outSize);
    auto copy = FFTWpp::vector<Real>(inSize);

    // Form the plans
    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure);

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
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure, kind);
    auto planBackward = Ranges::Plan(Ranges::View(out), Ranges::View(copy),
                                     Measure, kind.Inverse());

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
}