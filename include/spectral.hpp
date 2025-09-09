
#pragma once

#include <cmath>
#include <config.hpp>
#include <fftw3.h>
#include <omp.h>
#include <tensor.hpp>

const double pi = M_PI;

/** Spectral Solver class */
class SpectralSolver {

  public:
    
    int N;
    double L;
    double kmax;
    double* wavenumber = new double[N];
    double* laplacian = new double[N * N * N];
    double* Linv = new double[N * N * N];
    Params p;

    SpectralSolver(int N, double L, Params p, int nthreads); //constructor
    ~SpectralSolver();                                       // destructor

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

    // Helmholtz operator
    double* helmholtzOperator(double D, double dt);

  private:

    fftw_plan fft3_plan, ifft3_plan;
    
  };
