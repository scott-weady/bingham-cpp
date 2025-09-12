
#pragma once

#include <omp.h>
#include <fftw3.h>
#include <type_traits>

#include <spectral.hpp>
#include <tensor.hpp>

/** First-order explicit Euler time integration
 * @param U Current state
 * @param F Right-hand side
 * @param dt Time step
 * @param solver Spectral solver object
 * @return Updated state
 */
template <typename T>
T& euler(T& U, T& F, double dt, SpectralSolver& solver){

  auto N3 = solver.N * solver.N * solver.N;
  auto dT = solver.p.dim.dT;

  auto Linv = solver.helmholtzOperator(dT, dt);

  auto U_h = solver.fft(U);
  auto F_h = solver.fft(F);

  // Get unique pointers
  auto ptrsU = tensor::collect_pointers(U_h);
  auto ptrsF = tensor::collect_pointers(F_h);

  // Loop over all pointers and apply ifft
  for(auto i = 0; i < ptrsU.size(); i++){
    auto ptrU = ptrsU[i];
    auto ptrF = ptrsF[i];
    #pragma omp parallel for
    for(auto idx = 0; idx < N3; idx++){
      for(auto k = 0; k < 2; k++){
        ptrU[idx][k] = Linv[idx] * (ptrU[idx][k] + dt * ptrF[idx][k]);
      }
    }
  }

  U = solver.ifft(U_h);
  F = solver.ifft(F_h);

  return U;

}

/** Second-order implicit-explicit BDF2 time integration
 * @param U Current state
 * @param Um1 Previous state
 * @param F Right-hand side
 * @param Fm1 Previous right-hand side
 * @param dt Time step
 * @param dtm1 Previous time step
 * @param solver Spectral solver object
 * @return Updated state
 */
template <typename T>
T& sbdf2(T& U, T& Um1, T& F, T& Fm1, double dt, double dtm1, SpectralSolver& solver){

  auto N3 = solver.N * solver.N * solver.N;
  auto dT = solver.p.dim.dT;
  
  // Time step ratio
  auto r = dt / dtm1;
  auto a = (1 + r) * (1 + r) / (1 + 2 * r);
  auto am1 = -r * r / (1 + 2 * r);
  auto b = (1 + r) * (1 + r) / (1 + 2 * r);
  auto bm1 = -r * (1 + r) / (1 + 2 * r);

  auto Linv = solver.helmholtzOperator((1 + r) / (1 + 2 * r) * dT, dt);

  auto U_h = solver.fft(U);
  auto Um1_h = solver.fft(Um1);
  auto F_h = solver.fft(F);
  auto Fm1_h = solver.fft(Fm1);

  // Get unique pointers
  auto ptrsU = tensor::collect_pointers(U_h);
  auto ptrsUm1 = tensor::collect_pointers(Um1_h);
  auto ptrsF = tensor::collect_pointers(F_h);
  auto ptrsFm1 = tensor::collect_pointers(Fm1_h);

  // Loop over all pointers and apply sbdf2 step
  for(auto i = 0; i < ptrsU.size(); i++){

    auto ptrU = ptrsU[i];
    auto ptrUm1 = ptrsUm1[i];
    auto ptrF = ptrsF[i];
    auto ptrFm1 = ptrsFm1[i];
    
    #pragma omp parallel for
    for(auto idx = 0; idx < N3; idx++){
      for(auto k = 0; k < 2; k++){
        auto utemp = ptrU[idx][k];
        auto ftemp = ptrF[idx][k];
        ptrU[idx][k] = Linv[idx] * (a * ptrU[idx][k] + am1 * ptrUm1[idx][k] + dt * (b * ptrF[idx][k] + bm1 * ptrFm1[idx][k]));
        ptrUm1[idx][k] = utemp;
        ptrFm1[idx][k] = ftemp;
      }
    }
  }

  U = solver.ifft(U_h);
  Um1 = solver.ifft(Um1_h);
  F = solver.ifft(F_h);
  Fm1 = solver.ifft(Fm1_h);

  return U;

}