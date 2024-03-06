#error Documentation only

/**
@mainpage FFTWpp

@brief c++ interface to the fftw3 library.


    auto planForward =
        Ranges::Plan(Ranges::View(in), Ranges::View(out), Measure);
    auto planBackward =
        Ranges::Plan(Ranges::View(out), Ranges::View(copy), Measure);


*/

/** @namespace FFTWpp
    @brief Core functionality of the library.
*/

/** @namespace FFTWpp::Ranges
    @brief Additional functionality to allow for
    use of ranges in forming and executing plans.
*/
