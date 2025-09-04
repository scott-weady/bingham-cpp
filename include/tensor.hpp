
#pragma once

#include <array>
#include <fftw3.h>

namespace tensor{
        
    // Define tensor types
    using Tensor1 = std::array<fftw_complex*, 3>;
    using Tensor2 = std::array<Tensor1, 3>;
    using Tensor3 = std::array<Tensor2, 3>;

    // Initializers
    fftw_complex* zeros(int N);
    Tensor1 zeros1(int N);
    Tensor2 zeros2(int N);
    Tensor3 zeros3(int N);

}
