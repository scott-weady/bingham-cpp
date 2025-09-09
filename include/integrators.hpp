
#pragma once

#include <omp.h>
#include <fftw3.h>
#include <spectral.hpp>
#include <tensor.hpp>

template <typename T>
T& euler(T& U, T& F, double dt, SpectralSolver& solver, bool issymmetric){

  if constexpr (std::is_same_v<T, fftw_complex*>){

    auto N3 = solver.N * solver.N * solver.N;
    auto dT = solver.p.dim.dT;

    auto Linv = solver.helmholtzOperator(dT, dt);

    auto U_h = solver.fft(U, issymmetric);
    auto F_h = solver.fft(F, issymmetric);

    // Update, overwriting the current tensor
    #pragma omp parallel for
    for(auto idx = 0; idx < N3; idx++){
      for(auto k = 0; k < 2; k++){
        U_h[idx][k] = Linv[idx] * (U_h[idx][k] + dt * F_h[idx][k]);
      }
    }

    U = solver.ifft(U_h, issymmetric);
    F = solver.ifft(F_h, issymmetric);

    return U;
  } else {
    for(auto i = 0; i < U.size(); i++){
      U[i] = euler(U[i], F[i], dt, solver, issymmetric);
    }
    return U;
  }

}

template <typename T>
T& sbdf2(T& U, T& Um1, T& F, T& Fm1, double dt, double dtm1, SpectralSolver& solver, bool issymmetric){

  if constexpr (std::is_same_v<T, fftw_complex*>){

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

    #pragma omp parallel for
    for(auto idx = 0; idx < N3; idx++){
      for(auto k = 0; k < 2; k++){
        auto utemp = U_h[idx][k];
        auto ftemp = F_h[idx][k];
        U_h[idx][k] = Linv[idx] * (a * U_h[idx][k] + am1 * Um1_h[idx][k] + dt * (b * F_h[idx][k] + bm1 * Fm1_h[idx][k]));
        Um1_h[idx][k] = utemp;
        Fm1_h[idx][k] = ftemp;
      }
    }

    U = solver.ifft(U_h);
    Um1 = solver.ifft(Um1_h);
    F = solver.ifft(F_h);
    Fm1 = solver.ifft(Fm1_h);

    return U;

  } else {
    for(auto i = 0; i < U.size(); i++){
      U[i] = sbdf2(U[i], Um1[i], F[i], Fm1[i], dt, dtm1, solver, issymmetric);
    }
    return U;
  }

}