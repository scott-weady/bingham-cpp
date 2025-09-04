
#pragma once

#include <omp.h>
#include <random>

#include <spectral.hpp>
#include <tensor.hpp>

// Random integer
auto randi(int imax){
  std::uniform_int_distribution<int> dist(1, imax);
  std::random_device rd;
  std::mt19937 rng(42);
  return dist(rng);
}

// Random float on [0, 1]
auto randf(){
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::random_device rd;
  std::mt19937 rng(42);
  return dist(rng);
}

// Copy tensor fields
auto copy(tensor::Tensor1& dest, tensor::Tensor1& src){
  #pragma omp parallel for
  for(auto i = 0; i < 3; i++){
    for(auto idx = 0; idx < N * N * N; idx++){
      dest[i][idx][0] = src[i][idx][0];
      dest[i][idx][1] = src[i][idx][1];
    }
  }
};

// Copy tensor fields
auto copy(tensor::Tensor2& dest, tensor::Tensor2& src){
  #pragma omp parallel for
  for(auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      for(auto idx = 0; idx < N * N * N; idx++){
        dest[i][j][idx][0] = src[i][j][idx][0];
        dest[i][j][idx][1] = src[i][j][idx][1];
      }
    }
  }
};


// Pointwise magnitude of vector valued functions
auto magnitude(tensor::Tensor1 u, double* u_abs){

  // Loop over array
  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++)
    u_abs[idx] = std::sqrt(u[0][idx][0] * u[0][idx][0] + u[1][idx][0] * u[1][idx][0] + u[2][idx][0] * u[2][idx][0]);

} 

// L2 norm of vector valued functions over domain
auto L2(tensor::Tensor1 u, double dV){

  // Initialize
  auto u_norm = 0.0;

  // Loop over array
  #pragma omp parallel for reduction(+:u_norm)
  for(auto i = 0; i < 3; i++){
    for(auto idx = 0; idx < N * N * N; idx++) 
      u_norm += u[i][idx][0] * u[i][idx][0];
  }

  // Take square root and weight
  return std::sqrt(u_norm * dV);

} 

// L-infinity norm of vector valued functions over domain
auto Linf(tensor::Tensor1 u){

  // Initialize
  auto u_norm = 0.0;

  // Loop over array
  #pragma omp parallel for reduction(max:u_norm)
  for(auto i = 0; i < 3; i++){
    for(auto idx = 0; idx < N * N * N; idx++)
      u_norm = std::max(u_norm, std::abs(u[i][idx][0]));
  }

  return u_norm;

} 

// Generate a plane wave perturbation
auto planeWave(double* u, FastFourierTransform& fft){

  auto wavenumber = fft.wavenumber;
  auto N = fft.N;
  auto L = fft.L;

  // Wavenumber
  auto k1 = wavenumber[randi(4)];
  auto k2 = wavenumber[randi(4)];
  auto k3 = wavenumber[randi(4)];

  // Phase shift
  auto w1 = 2 * pi * randf();
  auto w2 = 2 * pi * randf();
  auto w3 = 2 * pi * randf();

  // Perturbation magnitude
  auto C = 0.001;

  // Fill in array
  for(auto nx = 0; nx < N; nx++){

    auto x = (double) nx * (L / N);

    for(auto ny = 0; ny < N; ny++){

      auto y = (double) ny * (L / N);

      for(auto nz = 0; nz < N; nz++){

        auto z = (double) nz * (L / N);

        auto idx = nz + N * ny + N * N * nx;
        
        u[idx] = C * std::cos(k1 * x + w1) * std::cos(k2 * y + w2) * std::cos(k3 * z + w3);

      }
    }
  }

  return u;

}

// Evaluate sum of plane wave perturbations
auto perturbation(fftw_complex* u, FastFourierTransform& fft){

  // Get resolution
  auto N = fft.N;

  // Number of perturbations
  auto Npert = 4;

  // Initialize as unique_ptr
  auto du = std::make_unique<double[]>(N * N * N);

  // Add perturbations
  for(auto n = 0; n < Npert; n++){
    planeWave(du.get(), fft);
    #pragma omp parallel for
    for(auto idx = 0; idx < N * N * N; idx++) u[idx][0] += du[idx];
  }

  return u;

}

// Enforce symmetry and unit trace
auto symmetrize(tensor::Tensor2 Q){

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    
    auto Q12 = Q[0][1][idx][0];
    auto Q13 = Q[0][2][idx][0];

    auto Q21 = Q[1][0][idx][0];
    auto Q23 = Q[1][2][idx][0];

    auto Q31 = Q[2][0][idx][0];
    auto Q32 = Q[2][1][idx][0];

    Q[0][1][idx][0] = 0.5 * (Q12 + Q21);
    Q[0][2][idx][0] = 0.5 * (Q13 + Q31);
    Q[1][2][idx][0] = 0.5 * (Q23 + Q32);
    Q[1][0][idx][0] = Q[0][1][idx][0];
    Q[2][0][idx][0] = Q[0][2][idx][0];
    Q[2][1][idx][0] = Q[1][2][idx][0];

    Q[2][2][idx][0] = 1.0 - Q[0][0][idx][0] - Q[1][1][idx][0];

  }

}