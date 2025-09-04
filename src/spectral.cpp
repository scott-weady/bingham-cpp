
#include <spectral.hpp>
#include <tensor.hpp>

FastFourierTransform::FastFourierTransform(int N, double L, int nthreads) : N(N), L(L) {

    fftw_plan_with_nthreads(nthreads);

    // Initialize arrays for FFT planner
    auto in = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));

    // Initialize plans
    fft3_plan = fftw_plan_dft_3d(N, N, N, in, in, -1, FFTW_MEASURE); // forward
    ifft3_plan = fftw_plan_dft_3d(N, N, N, in, in, 1, FFTW_MEASURE); // inverse

    // Free temporary array
    fftw_free(in);

    // Precompute Fourier modes
    for(int n = 0; n < N; n++){
        if(n <= N / 2) wavenumber[n] = 2.0 * pi * n / L;
        else           wavenumber[n] = 2.0 * pi * (n - N) / L;
    }
    
    // Maximum wave number
    kmax = (2.0 * pi  / L) * (N / 2);

    // Precompute laplacian
    #pragma omp parallel for
    for(auto nx = 0; nx < N; nx++){
        for(auto ny = 0; ny < N; ny++){
            for(auto nz = 0; nz < N; nz++){
                auto idx = nz + N * ny + N * N * nx;
                laplacian[idx] = -(wavenumber[nx] * wavenumber[nx] + wavenumber[ny] * wavenumber[ny] + wavenumber[nz] * wavenumber[nz]);
            }
        }
    }

}

// Destructor
FastFourierTransform::~FastFourierTransform() {
    fftw_destroy_plan(fft3_plan);
    fftw_destroy_plan(ifft3_plan);
    delete[] wavenumber;
    delete[] laplacian;
}

/**
 * Forward Fast Fourier Transform
 * 
 * Parameters:
 *    u (T) : input field
 * 
 * T can be nested std::array (Tensor1, Tensor2, Tensor3) or fftw_complex*
 * 
 * Example usage:
 *    u_h = fft(u);
 * 
 */
template <typename T>
T& FastFourierTransform::fft(T& u, bool issymmetric) {

    if constexpr (std::is_same_v<T, fftw_complex*>){
        fftw_execute_dft(fft3_plan, u, u);
    }
    else {
        for (auto& elem : u) fft(elem, issymmetric);  // recurse
    }
    
    return u;

}

// Instantiate template for types used
template fftw_complex*& FastFourierTransform::fft<fftw_complex*>(fftw_complex*&, bool);
template tensor::Tensor1& FastFourierTransform::fft<tensor::Tensor1>(tensor::Tensor1&, bool);
template tensor::Tensor2& FastFourierTransform::fft<tensor::Tensor2>(tensor::Tensor2&, bool);
template tensor::Tensor3& FastFourierTransform::fft<tensor::Tensor3>(tensor::Tensor3&, bool);

/**
 * Inverse Fast Fourier Transform
 * 
 * Parameters:
 *   u_h (T) : input field in Fourier space
 * 
 * T can be nested std::array (Tensor1, Tensor2, Tensor3) or fftw_complex*
 * 
 * Example usage:
 *   u = ifft(u_h);
 * 
 */
template <typename T>
T& FastFourierTransform::ifft(T& u, bool issymmetric) {

    if constexpr (std::is_same_v<T, fftw_complex*>){
        fftw_execute_dft(ifft3_plan, u, u);
        // Normalize
        #pragma omp parallel for
        for(auto idx = 0; idx < N * N * N; idx++){
            u[idx][0] /= (N * N * N);
            u[idx][1] /= (N * N * N);
        }
    }
    else {
        for (auto& elem : u) ifft(elem, issymmetric);  // recurse
    }
    
    return u;

}

// Instantiate template for types used
template fftw_complex*& FastFourierTransform::ifft<fftw_complex*>(fftw_complex*&, bool);
template tensor::Tensor1& FastFourierTransform::ifft<tensor::Tensor1>(tensor::Tensor1&, bool);
template tensor::Tensor2& FastFourierTransform::ifft<tensor::Tensor2>(tensor::Tensor2&, bool);
template tensor::Tensor3& FastFourierTransform::ifft<tensor::Tensor3>(tensor::Tensor3&, bool);

/**
 * Gradient operator
 * 
 * Parameters:
 *   u (T) : input field
 *  Du (T) : output gradient field
 * 
 * T can be nested std::array (Tensor1, Tensor2, Tensor3) or fftw_complex*
 * 
 * Example usage:
 *  Du = grad(u, Du);
 * 
 */
template <typename TensorIn, typename TensorOut>
TensorOut& FastFourierTransform::grad(TensorIn& u, TensorOut& Du){

    // Base case
    if constexpr (std::is_same_v<TensorIn, fftw_complex*>) {
    
    // Copy u into first component of gradu
    for(auto idx = 0; idx < N * N * N; idx++){
        Du[0][idx][0] = u[idx][0];
        Du[0][idx][1] = u[idx][1];
    }

    // Create reference to Fourier transform
    auto Du_h = fft(Du);

    // Loop over all points
    #pragma omp parallel for
    for(auto nx = 0; nx < N; nx++) {
        auto k1 = wavenumber[nx];
        for(auto ny = 0; ny < N; ny++) {
            auto k2 = wavenumber[ny];
            for(auto nz = 0; nz < N; nz++) {
                auto k3 = wavenumber[nz]; 
                
                // Compute flattened index for 3D array
                auto idx = nz + N * ny + N * N * nx;
                
                // Get real and imaginary parts
                auto u_h_real = Du_h[0][idx][0];
                auto u_h_imag = Du_h[0][idx][1];

                // Compute derivatives in each direction
                Du_h[0][idx][0] = -k1 * u_h_imag; // x (real part)
                Du_h[0][idx][1] =  k1 * u_h_real; // x (imaginary part)

                Du_h[1][idx][0] = -k2 * u_h_imag; // y (real part)
                Du_h[1][idx][1] =  k2 * u_h_real; // y (imaginary part)

                Du_h[2][idx][0] = -k3 * u_h_imag; // z (real part)
                Du_h[2][idx][1] =  k3 * u_h_real; // z (imaginary part)

            }
        } 
    }

    Du = ifft(Du_h);
    
    } 
    else {
        for (auto i = 0; i < u.size(); i++) grad(u[i], Du[i]);  // recurse
    }

    return Du;

}

// Instantiate template for types used
template tensor::Tensor1& FastFourierTransform::grad<fftw_complex*, tensor::Tensor1>(fftw_complex*&, tensor::Tensor1&);
template tensor::Tensor2& FastFourierTransform::grad<tensor::Tensor1, tensor::Tensor2>(tensor::Tensor1&, tensor::Tensor2&);
template tensor::Tensor3& FastFourierTransform::grad<tensor::Tensor2, tensor::Tensor3>(tensor::Tensor2&, tensor::Tensor3&);

/**
 * Anti-aliasing operator
 * 
 * Parameters:
 *   u (T) : input field
 * 
 * T can be nested std::array (Tensor1, Tensor2, Tensor3) or fftw_complex*
 * 
 * Example usage:
 *  u = antialias(u);
 * 
 */
template <typename T>
T& FastFourierTransform::antialias(T& u){

    if constexpr (std::is_same_v<T, fftw_complex*>) {

    auto u_h = fft(u);

    // Loop over all points
    #pragma omp parallel for
    for(auto nx = 0; nx < N; nx++){
        auto k1 = wavenumber[nx];
        for(auto ny = 0; ny < N; ny++){
            auto k2 = wavenumber[ny];
            for(auto nz = 0; nz < N; nz++){
                auto k3 = wavenumber[nz];

                if(std::abs(k1) > (2.0 / 3.0) * kmax || std::abs(k2) > (2.0 / 3.0) * kmax || std::abs(k3) > (2.0 / 3.0) * kmax){
                    auto idx = nz + N * ny + N * N * nx;
                    u_h[idx][0] = 0;
                    u_h[idx][1] = 0;
                }

            }
        }
    }

    // Inverse transform
    u = ifft(u_h);

    } else {
        for (auto& elem : u) elem = antialias(elem);
    }

    return u;

}

// Instantiate template for types used
template fftw_complex*& FastFourierTransform::antialias<fftw_complex*>(fftw_complex*&);
template tensor::Tensor1& FastFourierTransform::antialias<tensor::Tensor1>(tensor::Tensor1&);
template tensor::Tensor2& FastFourierTransform::antialias<tensor::Tensor2>(tensor::Tensor2&);
template tensor::Tensor3& FastFourierTransform::antialias<tensor::Tensor3>(tensor::Tensor3&);