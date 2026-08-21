#include <FFTWpp/Ranges>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

// This example shows how to use FFTW's wisdom: what to load at start-up,
// what to pre-generate, what happens when a plan's shape is not in the
// wisdom, and what to save at exit.
//
// Wisdom is what makes the thorough planner flags affordable. Measure and
// Patient time candidate algorithms on your actual arrays, which is slow;
// wisdom is FFTW's record of what it learned, so the second run of a program
// plans the same shapes almost instantly.

int main() {
  using namespace FFTWpp;

  using Real = double;
  using Complex = std::complex<Real>;

  // Wisdom is kept per precision, in a per-precision file format, so the
  // filename says which precision it holds.
  const auto wisdomFile =
      std::filesystem::temp_directory_path() / "fftwpp-example-double.wsd";
  std::filesystem::remove(wisdomFile);

  // Start from a known state so that the example demonstrates what it claims
  // to. A real program would not do this.
  ForgetWisdom();

  //--------------------------------------------------------------------//
  //           Loading whatever a previous run learned                  //
  //--------------------------------------------------------------------//
  {
    // On the first run there is no file yet, and that is not an error: it
    // simply means this run has to do the measuring. Import throws, so the
    // absence is handled rather than ignored.
    try {
      ImportWisdom<Real>(wisdomFile.string());
      std::cout << "Example5: loaded wisdom from a previous run\n";
    } catch (const std::runtime_error&) {
      std::cout << "Example5: no wisdom file yet, so this run will measure\n";
    }

    // A machine may also have system-wide wisdom, generated once by an
    // administrator. This reports rather than throws, because not having any
    // is the normal case.
    if (ImportSystemWisdom<Real>()) {
      std::cout << "Example5: system wisdom was available too\n";
    }
  }

  //--------------------------------------------------------------------//
  //             Pre-generating the shapes we will need                 //
  //--------------------------------------------------------------------//
  {
    // GenerateWisdom plans a shape without executing it, purely so that the
    // wisdom exists. Do this once at start-up for the shapes the program is
    // going to use, and the plans themselves cost nothing later.
    auto layout = Ranges::Layout(256);
    GenerateWisdom<Complex, Complex>(layout, layout, Measure);

    // The same for a real-to-complex pair, whose two sides differ in size.
    auto realLayout = Ranges::Layout(256);
    auto complexLayout = Ranges::Layout(256 / 2 + 1);
    GenerateWisdom<Real, Complex>(realLayout, complexLayout, Measure);

    std::cout << "Example5: generated wisdom for the shapes this run uses\n";
  }

  //--------------------------------------------------------------------//
  //                     Planning from the wisdom                       //
  //--------------------------------------------------------------------//
  {
    auto in = vector<Complex>(256);
    auto out = vector<Complex>(256);

    // WisdomOnly refuses to plan anything that is not already in the wisdom.
    // That makes it a useful assertion that the pre-generation above covered
    // what the program needs -- and a trap if it did not.
    auto plan =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), WisdomOnly, Forward);
    plan.Execute();
    std::cout << "Example5: planned a pre-generated shape with WisdomOnly\n";

    // A shape that was never generated is absent from the wisdom. With an
    // ordinary flag FFTW would simply plan it from scratch, and the result
    // would join the wisdom for next time. With WisdomOnly it fails, and
    // FFTWpp turns FFTW's null plan into an exception.
    auto oddIn = vector<Complex>(257);
    auto oddOut = vector<Complex>(257);
    try {
      [[maybe_unused]] auto missing = Ranges::Plan(
          Ranges::View(oddIn), Ranges::View(oddOut), WisdomOnly, Forward);
      std::cerr << "Example5: WisdomOnly planned a shape it should not have\n";
      return EXIT_FAILURE;
    } catch (const std::runtime_error& error) {
      std::cout << "Example5: as expected, WisdomOnly refused an unknown "
                   "shape -- "
                << error.what() << '\n';
    }

    // Without WisdomOnly the same shape plans perfectly well, and what FFTW
    // learns doing so is added to the wisdom.
    auto planned = Ranges::Plan(Ranges::View(oddIn), Ranges::View(oddOut),
                                Measure, Forward);
    planned.Execute();
    std::cout << "Example5: the same shape planned normally and joined the "
                 "wisdom\n";
  }

  //--------------------------------------------------------------------//
  //                    Keeping what this run learned                   //
  //--------------------------------------------------------------------//
  {
    // At exit, write the wisdom out so that the next run starts where this
    // one finished.
    ExportWisdom<Real>(wisdomFile.string());
    std::cout << "Example5: wrote wisdom to " << wisdomFile << '\n';

    // Wisdom also serialises to a string, for storing somewhere that is not
    // a file. Round-tripping it through ForgetWisdom shows that the string
    // really does carry everything.
    const auto serialised = ExportWisdomToString<Real>();
    ForgetWisdom<Real>();
    ImportWisdomFromString<Real>(serialised);

    auto in = vector<Complex>(256);
    auto out = vector<Complex>(256);
    auto plan =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), WisdomOnly, Forward);
    std::cout << "Example5: the string round trip preserved the wisdom\n";
  }

  std::filesystem::remove(wisdomFile);

  //--------------------------------------------------------------------//
  //                    Discarding it again, on purpose                 //
  //--------------------------------------------------------------------//
  {
    // CleanUp is the one call that interacts badly with wisdom: among other
    // things it discards it, so anything worth keeping must have been
    // exported already -- as it was, above.
    //
    // Most programs should not call CleanUp at all. FFTW's persistent state
    // is reachable for the life of the process, so leaving it is not a leak
    // and no leak checker reports one. It earns its keep only under a leak
    // checker that reports still-reachable blocks, in a plugin that may be
    // unloaded from a long-lived host, or when resetting FFTW deliberately --
    // which is what is being demonstrated here.
    //
    // Every plan must be destroyed first, since CleanUp leaves live ones
    // undefined. FFTWpp counts its own plans and throws rather than let that
    // happen silently.
    CleanUp();
    std::cout << "Example5: reset FFTW, discarding the wisdom in memory\n";

    // The file survives, so the next run starts where this one finished.
    ImportWisdomFromString<Real>(ExportWisdomToString<Real>());
  }

  std::cout << "Example5: the wisdom cycle completed\n";
  return EXIT_SUCCESS;
}
