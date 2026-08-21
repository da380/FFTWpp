// Exercises enough of the public interface, through the installed headers
// only, to prove that find_package(FFTWpp) yields a usable target.
#include <FFTWpp/Ranges>
#include <complex>
#include <cstdlib>
#include <iostream>

int main() {
  using Complex = std::complex<double>;

  constexpr int n = 16;
  auto in = FFTWpp::vector<Complex>(n);
  auto out = FFTWpp::vector<Complex>(n);
  auto back = FFTWpp::vector<Complex>(n);

  FFTWpp::RandomiseValues(in, std::uint64_t{20260821});
  const auto original = in;

  auto forward =
      FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(in), FFTWpp::Ranges::View(out),
                           FFTWpp::Estimate, FFTWpp::Forward);
  auto backward = FFTWpp::Ranges::Plan(FFTWpp::Ranges::View(out),
                                       FFTWpp::Ranges::View(back),
                                       FFTWpp::Estimate, FFTWpp::Backward);
  forward.Execute();
  backward.Execute();

  if (!FFTWpp::CheckValues(original, back, backward.Normalisation())) {
    std::cerr << "round trip through the installed package did not match\n";
    return EXIT_FAILURE;
  }

  std::cout << "FFTWpp package test passed"
            << (FFTWpp::ThreadsEnabled ? " (threads enabled)" : "") << '\n';
  FFTWpp::CleanUp();
  return EXIT_SUCCESS;
}
