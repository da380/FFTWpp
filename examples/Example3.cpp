
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

These additional arguments are including using variadic
templates. With the dimensions the number of arguments provides
the rank. With the real kinds, if fewer arguments are provided,
it is assumed that all remaining values are equal to the final
one provided. In particular, if you want all real kinds to be
the same, then only one value need be given.

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
    // Set dimension2.
    auto n0 = 10;
    auto n1 = 40;
    auto n2 = 5;

    //  Get data sizes.
    auto [inSize, outSize] = FFTWpp::DataSize<Real, Complex>(n0, n1, n2);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Real>(inSize);
    auto out = FFTWpp::vector<Complex>(outSize);
    auto copy = FFTWpp::vector<Real>(inSize);

    // Form the plans. Note that for out the final dimension is n2/2+1
    // due to in being real-valued.
    auto planForward =
        Ranges::Plan(Ranges::View(in, n0, n1, n2),
                     Ranges::View(out, n0, n1, n2 / 2 + 1), Measure);
    auto planBackward = Ranges::Plan(Ranges::View(out, n0, n1, n2 / 2 + 1),
                                     Ranges::View(copy, n0, n1, n2), Measure);

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
  //         4D real-real transformation pair       //
  //------------------------------------------------//
  {
    // Set dimension.
    auto n0 = 10;
    auto n1 = 40;
    auto n2 = 5;
    auto n3 = 5;

    //  Get data sizes.
    auto [inSize, outSize] = FFTWpp::DataSize<Real, Real>(n0, n1, n2, n3);

    // Allocate in, out, and copy arrays.
    auto in = FFTWpp::vector<Real>(inSize);
    auto out = FFTWpp::vector<Real>(outSize);
    auto copy = FFTWpp::vector<Real>(inSize);

    // Set the transform kind.
    auto kind = R2HC;

    // Form the plans. For the inverse transform we only need provide two
    // real kinds because from the second onward all the kinds are equal.
    auto planForward = Ranges::Plan(Ranges::View(in, n0, n1, n2, n3),
                                    Ranges::View(out, n0, n1, n2, n3), Measure,
                                    R2HC, DHT, DHT, DHT);
    auto planBackward = Ranges::Plan(Ranges::View(out, n0, n1, n2, n3),
                                     Ranges::View(copy, n0, n1, n2, n3),
                                     Measure, R2HC.Inverse(), DHT.Inverse());

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