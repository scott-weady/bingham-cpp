
#include <tensor.hpp>

/** Tensor initialization */
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

tensor::Tensor2 tensor::zeros2(int N, bool symmetric){
    tensor::Tensor2 u;

    if(!symmetric) for(auto i = 0; i < 3; i++) u[i] = tensor::zeros1(N); //if not symmetric
    else{
        for(auto i = 0; i < 3; i++){
            for(auto j = i; j < 3; j++){
                u[i][j] = tensor::zeros(N);
                u[j][i] = u[i][j]; //enforce symmetry
            }
        }
    }

    return u;
}

tensor::Tensor3 tensor::zeros3(int N, bool symmetric){

    tensor::Tensor3 u;
    
    if(!symmetric) for(auto i = 0; i < 3; i++) u[i] = tensor::zeros2(N);
    else{
        for(auto i = 0; i < 3; i++){
            for(auto j = i; j < 3; j++){
                for(auto k = 0; k < 3; k++){
                    u[i][j][k] = tensor::zeros(N);
                    u[j][i][k] = u[i][j][k]; //enforce symmetry
                }
            }
        }
    }

    return u;
}
