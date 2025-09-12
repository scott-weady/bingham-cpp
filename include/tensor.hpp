
#pragma once

#include <array>
#include <cstring>
#include <fftw3.h>
#include <type_traits>
#include <unordered_set>
#include <vector>

/** Tensor namespace */
namespace tensor{
        
    // Define tensor types
    using Tensor1 = std::array<fftw_complex*, 3>;
    using Tensor2 = std::array<Tensor1, 3>;
    using Tensor3 = std::array<Tensor2, 3>;

    // Initializers
    fftw_complex* zeros(int N);
    Tensor1 zeros1(int N);
    Tensor2 zeros2(int N, bool symmetric=false);
    Tensor3 zeros3(int N, bool symmetric=false);

    template <typename T>
    void collect_pointers_impl(const T& container,
                            std::unordered_set<fftw_complex*>& seen,
                            std::vector<fftw_complex*>& result) {

        if constexpr (std::is_same_v<T, fftw_complex*>) {
            if (seen.insert(container).second) result.push_back(container);
        } else {
            for (const auto& elem : container) collect_pointers_impl(elem, seen, result);
        }

    }

    template <typename T>
    std::vector<fftw_complex*> collect_pointers(const T& container) {
        std::unordered_set<fftw_complex*> seen;
        std::vector<fftw_complex*> result;
        collect_pointers_impl(container, seen, result);
        return result;
    }

}
