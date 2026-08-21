#include <FFTWpp/Ranges>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

// This example shows the guru interface: what it is for, and how to reach it
// without deriving strides by hand.
//
// Ranges::Layout describes repeated transforms with a single (howMany, dist)
// pair. That covers a great deal, but not everything. The guru interface gives
// every dimension its own input and output stride, and makes the loop over
// repetitions multi-dimensional as well.

int main() {
  using namespace FFTWpp;

  using Real = double;
  using Complex = std::complex<Real>;

  //--------------------------------------------------------------------//
  //              The case the advanced interface cannot reach          //
  //--------------------------------------------------------------------//
  {
    // A row-major array of shape (n0, n1, n2), to be transformed along its
    // middle axis. The transforms start at offsets i0 * n1 * n2 + i2, which
    // is not an arithmetic progression, so no single (howMany, dist) pair
    // describes them: with the advanced interface this needs n0 plans, or one
    // plan executed n0 times on different offsets.
    auto n0 = 3, n1 = 8, n2 = 5;

    // TransformAlong works the strides out from the shape and the axis. The
    // axes not listed become the repetition loop.
    auto layout = Ranges::TransformAlong({n0, n1, n2}, {1});

    std::cout << "06-guru_layouts: transform rank " << layout.Rank()
              << ", repetition rank " << layout.BatchRank() << ", "
              << layout.HowMany() << " transforms of length "
              << layout.TransformSize() << '\n';

    auto in = vector<Complex>(n0 * n1 * n2);
    auto out = vector<Complex>(n0 * n1 * n2);
    auto copy = vector<Complex>(n0 * n1 * n2);

    auto planForward = Ranges::GuruPlan(in, out, layout, Measure, Forward);
    auto planBackward = Ranges::GuruPlan(out, copy, layout, Measure, Backward);

    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();

    if (!CheckValues(in, copy, planBackward.Normalisation())) {
      std::cerr << "06-guru_layouts: the interior-axis round trip did not "
                   "match\n";
      return EXIT_FAILURE;
    }
  }

  //--------------------------------------------------------------------//
  //                    Several axes at once                            //
  //--------------------------------------------------------------------//
  {
    // Listing more axes transforms them jointly, as one multi-dimensional
    // transform, with whatever is left over as the repetition loop. Here axes
    // 0 and 2 are transformed and axis 1 is repeated over.
    auto shape = std::vector<std::ptrdiff_t>{4, 3, 6};
    auto layout = Ranges::TransformAlong(shape, {0, 2});

    auto size = shape[0] * shape[1] * shape[2];
    auto in = vector<Complex>(size);
    auto out = vector<Complex>(size);
    auto copy = vector<Complex>(size);

    auto planForward = Ranges::GuruPlan(in, out, layout, Measure, Forward);
    auto planBackward = Ranges::GuruPlan(out, copy, layout, Measure, Backward);

    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();

    // The normalisation is the product of the transformed extents only, so
    // 4 * 6 rather than 4 * 3 * 6.
    if (!CheckValues(in, copy, planBackward.Normalisation())) {
      std::cerr << "06-guru_layouts: the two-axis round trip did not match\n";
      return EXIT_FAILURE;
    }
    std::cout << "06-guru_layouts: normalisation over axes 0 and 2 is 1 / "
              << shape[0] * shape[2] << '\n';
  }

  //--------------------------------------------------------------------//
  //                    Real to complex, and back                       //
  //--------------------------------------------------------------------//
  {
    // The halfcomplex side is shorter along the last transformed axis. Both
    // builders take the *real* shape and work the other one out, and
    // DataSize reports how much storage each side needs.
    auto shape = std::vector<std::ptrdiff_t>{3, 8, 5};
    auto axes = std::vector<int>{1};

    auto forwardLayout = Ranges::RealToComplexTransformAlong(shape, axes);
    auto backwardLayout = Ranges::ComplexToRealTransformAlong(shape, axes);

    auto [realSize, complexSize] = DataSize<Real, Complex>(forwardLayout);
    std::cout << "06-guru_layouts: (3, 8, 5) along axis 1 needs " << realSize
              << " reals and " << complexSize << " complex numbers\n";

    auto in = vector<Real>(realSize);
    auto out = vector<Complex>(complexSize);
    auto copy = vector<Real>(realSize);

    auto planForward = Ranges::GuruPlan(in, out, forwardLayout, Measure);
    auto planBackward = Ranges::GuruPlan(out, copy, backwardLayout, Measure);

    RandomiseValues(in);
    planForward.Execute();
    planBackward.Execute();

    if (!CheckValues(in, copy, planBackward.Normalisation())) {
      std::cerr << "06-guru_layouts: the real-to-complex round trip did not "
                   "match\n";
      return EXIT_FAILURE;
    }
  }

  //--------------------------------------------------------------------//
  //               Writing a descriptor out by hand                     //
  //--------------------------------------------------------------------//
  {
    // For anything the builders do not cover, write the dimensions out. Here
    // the rows of a row-major array are transformed into a column-major
    // output: the two sides have different stride orders, which the advanced
    // interface cannot describe at all.
    //
    // Naming the fields is what keeps this readable: the two strides cannot
    // be transposed by accident, and the two dimension lists cannot be passed
    // the wrong way round because they are members of one object.
    auto rows = 4, columns = 6;
    auto layout =
        Ranges::GuruLayout{{{.n = columns, .inStride = 1, .outStride = rows}},
                           {{.n = rows, .inStride = columns, .outStride = 1}}};

    auto in = vector<Complex>(rows * columns);
    auto transposed = vector<Complex>(rows * columns);
    auto plan = Ranges::GuruPlan(in, transposed, layout, Estimate, Forward);
    RandomiseValues(in);
    plan.Execute();

    // The same transform without the transposition, to compare against.
    auto straight = vector<Complex>(rows * columns);
    auto plain = Ranges::Layout(1, std::vector{columns}, rows,
                                std::vector{columns}, 1, columns);
    auto reference =
        Ranges::Plan(Ranges::View(in, plain), Ranges::View(straight, plain),
                     Estimate, Forward);
    reference.Execute();

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        if (transposed[c * rows + r] != straight[r * columns + c]) {
          std::cerr << "06-guru_layouts: the transposing layout did not "
                       "match\n";
          return EXIT_FAILURE;
        }
      }
    }
    std::cout << "06-guru_layouts: a transposing layout matched the plain "
                 "transform\n";
  }

  //--------------------------------------------------------------------//
  //                     Layouts that cannot work                       //
  //--------------------------------------------------------------------//
  {
    // FFTW requires the map from indices to memory offsets to be injective,
    // and does not check it: a layout that visits an address twice computes a
    // wrong answer in silence. GuruLayout checks it instead.
    //
    // Here two dimensions of extent two both step by one, so index (0, 1) and
    // index (1, 0) both land on element 1.
    auto overlapping =
        Ranges::GuruLayout{{{.n = 2, .inStride = 1, .outStride = 1}},
                           {{.n = 2, .inStride = 1, .outStride = 1}}};
    try {
      overlapping.ValidateNonOverlapping(Ranges::Side::Input);
      std::cerr << "06-guru_layouts: an overlapping layout was accepted\n";
      return EXIT_FAILURE;
    } catch (const std::invalid_argument& error) {
      std::cout << "06-guru_layouts: overlap rejected -- " << error.what()
                << '\n';
    }
  }

  std::cout << "06-guru_layouts: every round trip matched\n";

  return EXIT_SUCCESS;
}
