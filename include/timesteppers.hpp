
#pragma once

#include <cmath>
#include <fftw3.h>
#include <omp.h>

#include <spectral.hpp>
#include <tensor.hpp>

/************************************************************
  Inverse Helmholtz operator for time stepping
************************************************************/
auto helmholtzOperator(double* Operator, double c, FastFourierTransform& fft){

  auto laplacian = fft.laplacian;
  auto N = fft.N;

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++) Operator[idx] = 1.0 / (1.0 - c * laplacian[idx]);

}

/************************************************************
  Implicit-explicit Euler method
************************************************************/
auto euler(tensor::Tensor2 U, tensor::Tensor2 F, double* inverseLinearOperator, double dt, FastFourierTransform& fft, bool issymmetric=false){

  auto U_h = fft.fft(U, issymmetric);
  auto F_h = fft.fft(F, issymmetric);

  // Update, overwriting the current tensor
  #pragma omp parallel for
  for(auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      for(auto idx = 0; idx < N * N * N; idx++){

        auto Linv = inverseLinearOperator[idx];
        for(auto k = 0; k < 2; k++) U_h[i][j][idx][k] = Linv * (U_h[i][j][idx][k] + dt * F_h[i][j][idx][k]);

      }
    }
  }

  U = fft.ifft(U_h, issymmetric);
  F = fft.ifft(F_h, issymmetric);

  return U;

}

// Helper function
auto sbdf2(double U, double Um1, double F, double Fm1, double dt, double dtm1){

  auto r = dt / dtm1;

  auto a = (1 + r) * (1 + r) / (1 + 2 * r);
  auto am1 = -r * r / (1 + 2 * r);

  auto b = (1 + r) * (1 + r) / (1 + 2 * r);
  auto bm1 = -r * (1 + r) / (1 + 2 * r);

  return a * U + am1 * Um1 + dt * (b * F + bm1 * Fm1);

}

// Main function
auto sbdf2(tensor::Tensor2 U, tensor::Tensor2 Um1, tensor::Tensor2 F, tensor::Tensor2 Fm1, double* inverseLinearOperator, double dt, double dtm1, FastFourierTransform& fft, bool issymmetric=false){

  auto U_h = fft.fft(U, issymmetric);
  auto Um1_h = fft.fft(Um1, issymmetric);
  auto F_h = fft.fft(F, issymmetric);
  auto Fm1_h = fft.fft(Fm1, issymmetric);

  #pragma omp parallel for
  for(auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      for(auto idx = 0; idx < N * N * N; idx++){

        auto Linv = inverseLinearOperator[idx]; 

        for(auto k = 0; k < 2; k++){

          auto utemp = U_h[i][j][idx][k];
          auto ftemp = F_h[i][j][idx][k];

          U_h[i][j][idx][k] = Linv * sbdf2(U_h[i][j][idx][k], Um1_h[i][j][idx][k], F_h[i][j][idx][k], Fm1_h[i][j][idx][k], dt, dtm1);
          
          Um1_h[i][j][idx][k] = utemp;
          Fm1_h[i][j][idx][k] = ftemp;

        }
      }
    }
  }

  U = fft.ifft(U_h, issymmetric);
  Um1 = fft.ifft(Um1_h, issymmetric);
  F = fft.ifft(F_h, issymmetric);
  Fm1 = fft.ifft(Fm1_h, issymmetric);

  return U;

}
