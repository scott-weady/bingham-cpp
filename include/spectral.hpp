
#pragma once

#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include <tensor.hpp>

const double pi = M_PI;

/** Fast Fourier Transform class */
class FastFourierTransform {

  public:
    
    int N;
    double L;
    double kmax;
    double* wavenumber = new double[N];
    double* laplacian = new double[N * N * N];

    FastFourierTransform(int N, double L, int nthreads); //constructor
    ~FastFourierTransform(); //destructor

    // Forward FFT
    template <typename T>
    T& fft(T& u, bool=false);

    // Inverse FFT
    template <typename T>
    T& ifft(T& u_h, bool=false);

    // 2/3 rule for dealiasing
    template <typename T>
    T& antialias(T& u);

    // Gradient operator
    template <typename TensorIn, typename TensorOut>
    TensorOut& grad(TensorIn& u, TensorOut& Du);

  private:

    fftw_plan fft3_plan, ifft3_plan;
    
  };
