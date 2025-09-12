

#pragma once

#include <cmath>
#include <fftw3.h>
#include <omp.h>

#include <config.hpp>
#include <tensor.hpp>

const double pi = M_PI;

/** Spectral Solver class
 * @param N Grid resolution
 * @param L Domain size
 * @param p Simulation parameters
 * @param nthreads Number of OpenMP threads
 * @return SpectralSolver object
 * 
 * Provides methods for FFT, iFFT, gradient, and Helmholtz operator
 */
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
    T& fft(T& u);

    // Inverse FFT
    template <typename T>
    T& ifft(T& u_h);

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
