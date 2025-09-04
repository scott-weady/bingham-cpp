
#include <cstring>
#include <tensor.hpp>

fftw_complex* tensor::zeros(int N) {
    auto arr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * N);
    std::memset(arr, 0, sizeof(fftw_complex) * N * N * N);
    return arr;
}

tensor::Tensor1 tensor::zeros1(int N){
    tensor::Tensor1 u;
    for(auto i = 0; i < 3; i++) u[i] = tensor::zeros(N);
    return u;
}

tensor::Tensor2 tensor::zeros2(int N){
    tensor::Tensor2 u;
    for(auto i = 0; i < 3; i++) u[i] = tensor::zeros1(N);
    return u;
}

tensor::Tensor3 tensor::zeros3(int N){
    tensor::Tensor3 u;
    for(auto i = 0; i < 3; i++) u[i] = tensor::zeros2(N);
    return u;
}
