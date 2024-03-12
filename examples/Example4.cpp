
#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <iostream>
#include <ranges>
#include <vector>

// This example shows how fftw3's advanced interface can be used perform
// multiple transformations, or transformations on data with more complicated
// layouts in memory.

int main() {
  using namespace FFTWpp;

  // Precision can be set freely between float, double and long double.
  using Real = double;
  using Complex = std::complex<Real>;

  // Peform multiple 1D complex to complex transformations. Here data
  // for each transformation is stored within a contiguous block of the
  // appropriate size. This means that the offset of the ith datum for
  // the jth transformation is equal to i + n * j.
  {
    // Set the size of the data.
    auto n = 200;

    // Set the number of transformations.
    auto howMany = 10;

    // Define the layout parameters explicitly.
    auto rank = 1;                // Dimension of the data.
    auto sizes = std::vector{n};  // Size along each axis.
    auto embed = sizes;           // Embeded size along each axis.
    auto stride = 1;  // Distance between data within each transform.
    auto dist = n;  // Distance between start of data for different transforms.

    // Form the layout and use this to get the storage size. Note that
    // in this case the in and out layouts are identical.
    auto layout = Ranges::Layout(rank, sizes, howMany, embed, stride, dist);
    auto size = layout.size();

    // Allocate the data.
    auto in = vector<Complex>(size);
    auto out = vector<Complex>(size);
    auto copy = vector<Complex>(size);

    // Form the plans.
    auto planForward = Ranges::Plan(
        Ranges::View(in, layout), Ranges::View(out, layout), Measure, Forward);
    auto planBackward =
        Ranges::Plan(Ranges::View(out, layout), Ranges::View(copy, layout),
                     Measure, Backward);

    // Set values for in.
    FFTWpp::RandomiseValues(in);

    // Execute the plans.
    planForward.Execute();
    planBackward.Execute();

    // Check the transforms worked.
    if (!CheckValues(in, copy, planBackward.Normalisation())) {
      std::cout << "Transform not okay\n";
    }
  }
}