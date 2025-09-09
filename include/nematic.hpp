#pragma once

#include <bingham.hpp>
#include <spectral.hpp>
#include <tensor.hpp>

#include <omp.h>
#include <utils.hpp>

// Nonlinear term in Q-tensor equation
auto evaluateNonlinearity(tensor::Tensor2 F, tensor::Tensor1 u, tensor::Tensor2 Du, tensor::Tensor2 Q, tensor::Tensor3 DQ, BinghamClosure& closure, SpectralSolver& fft){ 

  fft.grad(u, Du);  //velocity gradient
  fft.grad(Q, DQ); //gradient of Q
  auto ST = closure.compute(Q, Du); //Bingham closure (returns reference to internal ST variable)
  
  // Main loop 
  #pragma omp parallel for 
  for(auto i = 0; i < 3; i++){ 
    for(auto j = 0; j < 3; j++){ 
      for(auto idx = 0; idx < N * N * N; idx++){ 

        auto u_dot_DQ = 0.0; 
        for(auto k = 0; k < 3; k++) u_dot_DQ += u[k][idx][0] * DQ[i][j][k][idx][0]; 
        
        auto QQ = 0.0; 
        for(auto k = 0; k < 3; k++) QQ += Q[i][k][idx][0] * Q[k][j][idx][0]; 

        auto QDU = 0.0;
        for(auto k = 0; k < 3; k++) QDU += Du[i][k][idx][0] * Q[k][j][idx][0] + Q[k][i][idx][0] * Du[j][k][idx][0]; 
        
        F[i][j][idx][0] = -u_dot_DQ + QDU - 2 * ST[i][j][idx][0] + 4 * zeta * QQ - 6 * dR * (Q[i][j][idx][0] - (i == j ? 1.0 / 3.0 : 0.0)); 

      }
    } 
  }

  fft.antialias(F); //dealiasing
  return F;

}

// Nematic stress tensor
auto stress(tensor::Tensor2 Sigma, tensor::Tensor2 Q, tensor::Tensor2 ST){

  #pragma omp parallel for
  for(auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      for(auto idx = 0; idx < N * N * N; idx++){

        auto QQ = 0.0;
        for(auto k = 0; k < 3; k++) QQ += Q[i][k][idx][0] * Q[k][j][idx][0];

        Sigma[i][j][idx][0] = sigma_a * Q[i][j][idx][0] + sigma_b * ST[i][j][idx][0] - 2 * sigma_b * zeta * QQ;

      }
    }
  }

  return Sigma;
  
}

// Spectral Stokes solver
auto StokesSolver(tensor::Tensor1 u, tensor::Tensor2 Sigma, SpectralSolver& fft){

  // Compute Fourier transform
  auto u_h = fft.fft(u);
  auto Sigma_h = fft.fft(Sigma, true); //true indicates tensor is symmetric
  auto wavenumber = fft.wavenumber;

  #pragma omp parallel for
  for(auto nx = 0; nx < N; nx++){

    // x wavenumber
    auto k1 = wavenumber[nx];

    for(auto ny = 0; ny < N; ny++){
      
      // y wavenumber
      auto k2 = wavenumber[ny];
      
      for(auto nz = 0; nz < N; nz++){

        // z wavenumber
        auto k3 = wavenumber[nz]; 

        // magnitude of wavevector
        auto ksq = k1 * k1 + k2 * k2 + k3 * k3;

        // Flattened index
        auto idx = nz + N * ny + N * N * nx;  
        
        // Stokes operator
        auto L11 = (1.0 / ksq) * (1.0 - k1 * k1 / ksq);
        auto L12 = (1.0 / ksq) * (0.0 - k1 * k2 / ksq);
        auto L13 = (1.0 / ksq) * (0.0 - k1 * k3 / ksq);
        auto L22 = (1.0 / ksq) * (1.0 - k2 * k2 / ksq);
        auto L23 = (1.0 / ksq) * (0.0 - k2 * k3 / ksq);
        auto L33 = (1.0 / ksq) * (1.0 - k3 * k3 / ksq);

        fftw_complex rhs1, rhs2, rhs3;
       
        // Evaluate divergence
        rhs1[0] = -(k1 * Sigma_h[0][0][idx][1] + k2 * Sigma_h[0][1][idx][1] + k3 * Sigma_h[0][2][idx][1]); 
        rhs2[0] = -(k1 * Sigma_h[1][0][idx][1] + k2 * Sigma_h[1][1][idx][1] + k3 * Sigma_h[1][2][idx][1]); 
        rhs3[0] = -(k1 * Sigma_h[2][0][idx][1] + k2 * Sigma_h[2][1][idx][1] + k3 * Sigma_h[2][2][idx][1]);  

        rhs1[1] = (k1 * Sigma_h[0][0][idx][0] + k2 * Sigma_h[0][1][idx][0] + k3 * Sigma_h[0][2][idx][0]);
        rhs2[1] = (k1 * Sigma_h[1][0][idx][0] + k2 * Sigma_h[1][1][idx][0] + k3 * Sigma_h[1][2][idx][0]); 
        rhs3[1] = (k1 * Sigma_h[2][0][idx][0] + k2 * Sigma_h[2][1][idx][0] + k3 * Sigma_h[2][2][idx][0]);

        for(auto k = 0; k < 2; k++){ 
          u_h[0][idx][k] = L11 * rhs1[k] + L12 * rhs2[k] + L13 * rhs3[k];
          u_h[1][idx][k] = L12 * rhs1[k] + L22 * rhs2[k] + L23 * rhs3[k];
          u_h[2][idx][k] = L13 * rhs1[k] + L23 * rhs2[k] + L33 * rhs3[k];
        }

      }
    }
  }

  // Set int(U) = 0
  u_h[0][0][0] = 0.0;
  u_h[0][0][1] = 0.0;

  u_h[1][0][0] = 0.0;
  u_h[1][0][1] = 0.0;

  u_h[2][0][0] = 0.0;
  u_h[2][0][1] = 0.0;

  // Convert to real space
  u = fft.ifft(u_h);
  Sigma = fft.ifft(Sigma_h, true);

  return u;

}

// Iterative fluid solver with nematic stress
auto fluidSolver(tensor::Tensor1 u, tensor::Tensor2 Du, tensor::Tensor2 Q, tensor::Tensor2 Sigma, BinghamClosure& ST, tensor::Tensor1 up1, SpectralSolver& fft, double tolerance=1e-8, int maxIterations=10){

  // Compute unconstrained stress Sigma = sigma_a * Q
  #pragma omp parallel for
  for (auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      for(auto idx = 0; idx < N * N * N; idx++) Sigma[i][j][idx][0] = sigma_a * Q[i][j][idx][0];
    }
  }

  // Solve for velocity field with unconstrained stress
  StokesSolver(u, Sigma, fft);

  if(sigma_b == 0) return u; //no need to iterate if sigma_b = 0
  
  // Set initial iteration counter
  auto iteration = 0;

  // Set initial error
  auto error = 10 * tolerance;
  
  while(error > tolerance && iteration < maxIterations){

    // Compute velocity gradient
    fft.grad(u, Du);

    // Update stress
    stress(Sigma, Q, ST.compute(Q, Du));

    // Solve for fluid velocity
    StokesSolver(up1, Sigma, fft);

    // Initialize current iteration error
    error = 0;

    // Check update
    #pragma omp parallel for reduction(+:error)
    for(auto i = 0; i < 3; i++){
      for(auto idx = 0; idx < N * N * N; idx++) error += std::abs(up1[i][idx][0] - u[i][idx][0]);
    }

    // Prepare for next loop
    std::swap(u, up1);

    iteration++;

  }

  return u;
  
}

// Compute nematic order parameter from Q-tensor
auto nematicOrderParameter(tensor::Tensor2 Q, fftw_complex* s){
  
  auto tolerance = 1e-14; //convergence tolerance for eig solve
  auto maxIterations = 100; //std::max iterations for eig solve

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    
    auto mu = (1.0 / 3.0); //initial guess

    // Local copy of the Q tensor::Tensor
    auto Q11 = Q[0][0][idx][0];
    auto Q12 = Q[0][1][idx][0] + 1e-15;
    auto Q13 = Q[0][2][idx][0] + 1e-15;
    auto Q22 = Q[1][1][idx][0];
    auto Q23 = Q[1][2][idx][0] + 1e-15;
    auto Q33 = Q[2][2][idx][0];

    // Coefficients of characteristic polynomial
    auto a0 = Q11 * Q23 * Q23 + Q22 * Q13 * Q13 + Q33 * Q12 * Q12 - Q11 * Q22 * Q33 - 2 * Q13 * Q12 * Q23;
    auto a1 = Q11 * Q22 + Q11 * Q33 + Q22 * Q33 - (Q12 * Q12 + Q13 * Q13 + Q23 * Q23);

    // Initial function evaluation
    auto chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;
 
    // Initialize iteration
    auto iteration = 0; //iteration count

    // Find root of characteristic polynomial using Newtons method
    while(std::abs(chi_mu) > tolerance && iteration < maxIterations){

      // Evaluate characteristic polynomial
      chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;

      // Update
      mu -= chi_mu / (3 * mu * mu - 2 * mu + a1);

      iteration++;

    }

    // Get other eigenvalues
    auto nu1 = mu;
    auto nu2 = 0.5 * (-(mu - 1) + std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    auto nu3 = 0.5 * (-(mu - 1) - std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    
    mu = std::max(std::max(nu1, nu2), nu3); //sort 
    s[idx][0] = 1.5 * (mu - 1.0 / 3); //store
    s[idx][1] = 0.0;
    
  }
}

// Generate or load initial condition
auto initialCondition(tensor::Tensor2 Q, bool resume, SpectralSolver& fft){

  try{
      
    if(!resume) throw std::runtime_error("No initial data requested.");

    std::ifstream Q_init("initial_data/Q/Q.dat");
    if(!Q_init) throw std::runtime_error("No initial data found.");

    std::cout << "Initial data found! Loading Q..." << '\n';

    double q11, q12, q13;
    double      q22, q23;
    double           q33;

    // Line counter
    auto idx = 0;

    // Load
    while (Q_init >> q11 >> q12 >> q13 >> q22 >> q23 >> q33) {

      // Assign values
      if(idx < N * N * N){
        Q[0][0][idx][0] = q11, Q[0][1][idx][0] = q12, Q[0][2][idx][0] = q13;
        Q[1][0][idx][0] = q12, Q[1][1][idx][0] = q22, Q[1][2][idx][0] = q23;
        Q[2][0][idx][0] = q13, Q[2][1][idx][0] = q23, Q[2][2][idx][0] = q33;
      }

      idx++;

    }

    // Check if correct dimensions
    if(idx != N * N * N){
      Q_init.close();
      throw std::runtime_error("Incompatible resolution (" + std::to_string(idx) + " points given, expected " + std::to_string(N * N * N) + ").");
    }

    Q_init.close();

  } catch(const std::runtime_error& e){

    std::cout << e.what() << " Generating new initial condition..." << '\n';

    for(auto i = 0; i < 3; i++){
      for(auto j = 0; j < 3; j++){

        perturbation(Q[i][j], fft);
        
        if(i == j){
          #pragma omp parallel for
          for(auto idx = 0; idx < N * N * N; idx++) Q[i][j][idx][0] += 1.0 / 3.0;
        }
      
      }
    }
  }

  symmetrize(Q);
  return Q;

}
