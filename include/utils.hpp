
#ifndef utils
#define utils

#include <random>

/* Creates a directory under "save" with name foldername */
auto createFolder(std::string foldername){

  std::string message = "mkdir " + foldername;
  int check = system(message.c_str());

}

// /************************************************************
//   Random integer up to imax
// ************************************************************/
// int randi(int imax){

//   return (1 + (rand() % imax));

// }

// /************************************************************
//   Random float between 0 and 1
// ************************************************************/
// double randf(){

//   return ((double) rand() / (RAND_MAX));

// }

/************************************************************
  Random integer up to imax
************************************************************/
auto randi(int imax){

  std::uniform_int_distribution<int> dist(1, imax);
  std::random_device rd;
  std::mt19937 rng(42);
  return dist(rng);

}

/************************************************************
  Random float between 0 and 1
************************************************************/
auto randf(){

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::random_device rd;
  std::mt19937 rng(42);
  return dist(rng);

}

/************************************************************
  Array of zeros
************************************************************/
auto zeros(int num_elements) {
  fftw_complex* arr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_elements);
  std::memset(arr, 0, sizeof(fftw_complex) * num_elements);
  return arr;
}

/************************************************************
  Fast Fourier Transform class
************************************************************/
class FastFourierTransform {

  public:
    
    fftw_plan fft3_plan, ifft3_plan;
    fftw_complex *in, *out;

    // Constructor
    FastFourierTransform(int N) {

      // Initialize arrays for FFT planner
      in = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));
      out = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex)); 

      // Forward transform
      fft3_plan = fftw_plan_dft_3d(N, N, N, in, out, -1, FFTW_ESTIMATE); 

      // Inverse transform
      ifft3_plan = fftw_plan_dft_3d(N, N, N, in, out, 1, FFTW_ESTIMATE);

    }

    ~FastFourierTransform() {
        fftw_destroy_plan(fft3_plan);
        fftw_destroy_plan(ifft3_plan);
        fftw_free(in);
        fftw_free(out);
    }

    auto fft(fftw_complex* u, fftw_complex* u_h){
      fftw_execute_dft(fft3_plan, u, u_h);
    }

    auto fft(fftw_complex* u[3], fftw_complex* u_h[3]){
      for(int i = 0; i < 3; i++){
        fft(u[i], u_h[i]);
      }
    }

    auto fft(fftw_complex* u[3][3], fftw_complex* u_h[3][3], bool symmetric=true){
      for(int i = 0; i < 3; i++){
        for(int j = (symmetric ? i : 0); j < 3; j++){
          fft(u[i][j], u_h[i][j]);
        }
      }
    }

    auto ifft(fftw_complex* u_h, fftw_complex* u){
      fftw_execute_dft(ifft3_plan, u_h, u);
    }

    auto ifft(fftw_complex* u_h[3], fftw_complex* u[3]){
      for(int i = 0; i < 3; i++){
        ifft(u_h[i], u[i]);
      }
    }

    auto ifft(fftw_complex* u_h[3][3], fftw_complex* u[3][3], bool symmetric=true){
      for(auto i = 0; i < 3; i++){
        for(auto j = (symmetric ? i : 0); j < 3; j++){
          ifft(u_h[i][j], u[i][j]);
        }
      }
    }

  };

/************************************************************
  Inverse Helmholtz operator for time stepping

  Parameters:

    Operator (double*) : array for storing operator, overwritten
    wavenumber (double*) : N x 1 array of Fourier modes
    c (double) : Helmholtz coefficient

************************************************************/
auto helmholtzOperator(double* Operator, double* wavenumber, double c){

  // Fill in operator
  #pragma omp parallel for
  for(int ix = 0; ix < N; ix++){
    for(int iy = 0; iy < N; iy++){
      for(int iz = 0; iz < N; iz++){

        // Flattened index
        auto idx = iz + N * iy + N * N * ix;

        // Laplacian
        double laplacian = -(wavenumber[ix] * wavenumber[ix] + wavenumber[iy] * wavenumber[iy] + wavenumber[iz] * wavenumber[iz]);

        // Inverse Euler operator        
        Operator[idx] = 1.0 / (1.0 - c * laplacian) * inv_Z;
        
      }
    }
  }

}

/************************************************************  
  Gradient operator for rank 0, 1, and 2 tensors (Fourier)

  Parameters:

    fft3 (fftw_plan): Forward FFT plan
    ifft3 (fftw_plan): Inverse FFT plan
    U (Vector, Tensor): input variable in real space
    gradU (Vector, Tensor, Tensor3) : gradient of input in real space

************************************************************/

// Helper function
auto grad(fftw_complex* u_h, fftw_complex* gradu_h[3], double* wavenumber){

  // Loop over all points
  #pragma omp parallel for
  for(int ix = 0; ix < N; ix++) {
    
    double k1 = wavenumber[ix];
    
    for(int iy = 0; iy < N; iy++) {
      
      double k2 = wavenumber[iy];
      
      for(int iz = 0; iz < N; iz++) {
      
        double k3 = wavenumber[iz]; 
        
        // Compute flattened index for 3D array
        auto idx = iz + N * iy + N * N * ix;
        
        // Get real and imaginary parts
        double u_h_real = u_h[idx][0];
        double u_h_imag = u_h[idx][1];

        // Compute derivatives in each direction
        gradu_h[0][idx][0] = -k1 * u_h_imag * inv_Z; // x (real part)
        gradu_h[0][idx][1] =  k1 * u_h_real * inv_Z; // x (imaginary part)

        gradu_h[1][idx][0] = -k2 * u_h_imag * inv_Z; // y (real part)
        gradu_h[1][idx][1] =  k2 * u_h_real * inv_Z; // y (imaginary part)

        gradu_h[2][idx][0] = -k3 * u_h_imag * inv_Z; // z (real part)
        gradu_h[2][idx][1] =  k3 * u_h_real * inv_Z; // z (imaginary part)

      }
    } 
  }
}

// Vector, convention is grad(U)_ij = dU_i / dx_j
auto grad(fftw_complex* u[3], fftw_complex* u_h[3], fftw_complex* gradU[3][3], fftw_complex* buffer[3], double* wavenumber, FastFourierTransform& fft){

  // Compute Fourier transform
  fft.fft(u, u_h);

  // Compute the gradient of each component
  for(auto i = 0; i < 3; i++){
    grad(u_h[i], buffer, wavenumber);
    fft.ifft(buffer[0], gradU[i][0]);
    fft.ifft(buffer[1], gradU[i][1]);
    fft.ifft(buffer[2], gradU[i][2]);
  }

}


// Rank-2 tensor (assumed symmetric), convention is grad(U)_ijk = dU_ij / dx_k
auto grad(fftw_complex* U[3][3], fftw_complex* U_h[3][3], fftw_complex* gradU[3][3][3], fftw_complex* buffer[3], double* wavenumber, FastFourierTransform& fft){

  // Compute Fourier transform
  fft.fft(U, U_h);

  // Compute the gradient of each component
  for(auto i = 0; i < 3; i++){
    for(auto j = i; j < 3; j++){

      grad(U_h[i][j], buffer, wavenumber);
      
      fft.ifft(buffer[0], gradU[i][j][0]);
      fft.ifft(buffer[1], gradU[i][j][1]);
      fft.ifft(buffer[2], gradU[i][j][2]);

      if(i != j){
        gradU[j][i][0] = gradU[i][j][0];
        gradU[j][i][1] = gradU[i][j][1];
        gradU[j][i][2] = gradU[i][j][2];
      }
    }

  }
  
}

/************************************************************
  Anti-aliasing via the 2 / 3 rule
  
  Parameters:

    U_h (Tensor) : field in Fourier space, overwritten
    wavenumber (double*) : N x 1 array of Fourier modes
    kmax (double) : maximum wave number

************************************************************/
auto antialias(fftw_complex* U_h[3][3], double* wavenumber, double kmax){

  // Loop over all points
  #pragma omp parallel for
  for(auto ix = 0; ix < N; ix++){

    double k1 = wavenumber[ix];

    for(auto iy = 0; iy < N; iy++){

      double k2 = wavenumber[iy];

      for(auto iz = 0; iz < N; iz++){

        double k3 = wavenumber[iz];

        if(std::abs(k1) > (2.0 / 3.0) * kmax || 
           std::abs(k2) > (2.0 / 3.0) * kmax || 
           std::abs(k3) > (2.0 / 3.0) * kmax){

          // Flattened index
          auto idx = iz + N * iy + N * N * ix;

          for(auto i = 0; i < 3; i++)
            for(auto j = 0; j < 3; j++){
              U_h[i][j][idx][0] = 0;
              U_h[i][j][idx][1] = 0;

            }
          }
          

      }
    }
  }

}

/************************************************************
  Implicit-explicit Euler method, assuming diagonalized system of ODEs

  Parameters:

    U (Tensor): solution at current time, overwritten
    F (Tensor): explicit terms in ODE
    inverseLinearOperator (double*) : inverse of implicit discrete operator
    dt (double) : time step size

************************************************************/

// Helper function
double euler(double U, double F, double dt){

  return U + dt * F;

}

// Main function
auto euler(fftw_complex* U[3][3], fftw_complex* F[3][3], double* inverseLinearOperator, double dt){

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){

    double Linv = inverseLinearOperator[idx];

    // Update, overwriting the current tensor
    for(auto i = 0; i < 3; i++){
      for(auto j = i; j < 3; j++){
        for(auto k = 0; k < 2; k++){
          U[i][j][idx][k] = Linv * euler(U[i][j][idx][k], F[i][j][idx][k], dt);
        }
      }
    }

  }

}

/************************************************************
  Implicit-explicit Euler method, assuming diagonalized system of ODEs

  Parameters:

    U (Tensor): solution at current time, overwritten
    Um1 (Tensor): solution at previous time, overwritten
    F (Tensor): explicit terms in ODE at current time
    Fm1 (Tensor): explicit terms in ODE at previous time, overwritten
    inverseLinearOperator (double*) : inverse of implicit discrete operator
    dt (double) : time step size

************************************************************/


// Helper function
double sbdf2(double U, double Um1, double F, double Fm1, double dt, double dtm1){

  double r = dt / dtm1;

  double a = (1 + r) * (1 + r) / (1 + 2 * r);
  double am1 = -r * r / (1 + 2 * r);

  double b = (1 + r) * (1 + r) / (1 + 2 * r);
  double bm1 = -r * (1 + r) / (1 + 2 * r);

  return a * U + am1 * Um1 + dt * (b * F + bm1 * Fm1);


}

// Main function
auto sbdf2(fftw_complex* U[3][3], fftw_complex* Um1[3][3], fftw_complex* F[3][3], fftw_complex* Fm1[3][3], double* inverseLinearOperator, double dt, double dtm1){

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){

    double Linv = inverseLinearOperator[idx]; 

    for(auto i = 0; i < 3; i++){
      for(auto j = i; j < 3; j++){
        for(auto k = 0; k < 2; k++){

          double utemp = U[i][j][idx][k];
          double ftemp = F[i][j][idx][k];

          U[i][j][idx][k] = Linv * sbdf2(U[i][j][idx][k], Um1[i][j][idx][k], F[i][j][idx][k], Fm1[i][j][idx][k], dt, dtm1);
          Um1[i][j][idx][k] = utemp;
          Fm1[i][j][idx][k] = ftemp;

        }
      }
    }
  
  }

}

/************************************************************
  Nonlinear/nonstiff part of the dQ/dt equation

  Mathematically, this function evaluates the tensor-valued function

    F = -U * grad(Q) + (T * Q + Q * T') - 2 * S * T - 6 * dR * (Q - I / 3),

  where

    T = grad(U) + 2 * zeta * Q

  is the rotational term in the orientation dynamics.

  Parameters:

    F (Tensor): output
    U (Vector): velocity field
    gradU (Tensor): velocity gradient
    Q (Tensor): nematic tensor
    gradQ (Tensor3) : gradient of the nematic tensor
    ST (Tensor): contraction of the 4th moment tensor S : T

************************************************************/
auto nonlinear(fftw_complex* F[3][3], fftw_complex* u[3], fftw_complex* gradU[3][3], fftw_complex* Q[3][3], fftw_complex* gradQ[3][3][3], fftw_complex* ST[3][3]){ 
  
  // Main loop 
  #pragma omp parallel for 
  for(auto idx = 0; idx < N * N * N; idx++){ 

    for(auto i = 0; i < 3; i++){ 
      for(auto j = 0; j < 3; j++){ 

        double u_dot_gradQ = 0; 

        for(auto k = 0; k < 3; k++){ 
          u_dot_gradQ += u[k][idx][0] * gradQ[i][j][k][idx][0]; 
        } 
        
        double QQ = 0; 
        
        for(auto k = 0; k < 3; k++){ 
          QQ += Q[i][k][idx][0] * Q[k][j][idx][0]; 
        }

        double QDU = 0;

        for(auto k = 0; k < 3; k++){ 
          QDU += gradU[i][k][idx][0] * Q[k][j][idx][0] + Q[k][i][idx][0] * gradU[j][k][idx][0]; 
        }
        
        F[i][j][idx][0] = -u_dot_gradQ + QDU - 2 * ST[i][j][idx][0] + 4 * zeta * QQ - 6 * dR * (Q[i][j][idx][0] - (i == j ? 1.0 / 3.0 : 0.0)); 
      
      } 
    } 

  }
}

/************************************************************
  Stress tensor
    
  Mathematically, this function evaluates the tensor-valued function

    Sigma = sigma_a * Q + sigma_b * S : E + 2 * sigma_b * zeta * (S : Q - Q * Q)

  Parameters: 

    Sigma (Tensor): stress tensor, output
    Q (Tensor): nematic tensor
    ST (Tensor): contraction of the 4th moment tensor S : T

************************************************************/
auto stress(fftw_complex* Sigma[3][3], fftw_complex* Q[3][3], fftw_complex* ST[3][3]){

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
     
    for(auto i = 0; i < 3; i++){
      for(auto j = i; j < 3; j++){

        double QQ = 0;

        for(auto k = 0; k < 3; k++){
          QQ += Q[i][k][idx][0] * Q[k][j][idx][0];
        }

        Sigma[i][j][idx][0] = sigma_a * Q[i][j][idx][0] + sigma_b * ST[i][j][idx][0] - 2 * sigma_b * zeta * QQ;
      }
    }
  }
  
}

/************************************************************
  Bingham closure

  This function evaluates the contraction of the 4th moment tensor S = <pppp> with
  the rotation tensor T = grad(U) + 2 * zeta * Q using the Bingham closure. The degree
  of the Chebyshev interpolants is specified in the params.h file.

  Parameters:

    ST (Tensor) : contraction of the 4th moment S : T
    Q (Tensor) : nematic tensor
    gradU (Tensor): velocity gradient

************************************************************/
auto binghamClosure(fftw_complex* ST[3][3], fftw_complex* Q[3][3], fftw_complex* gradU[3][3]){

  // Tolerance for eigenvalue solve
  double tolerance = 1e-14;

  // Max iterations for eigenvalue solve 
  int maxIterations = 50; 

  // Loop over all grid points
  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    
    // Initialize rotated tensor components
    double tS1111, tS1122, tS1133;
    double         tS2222, tS2233;
    double                 tS3333; 

    // Rotated contraction S:T
    double tST11, tST12, tST13;
    double        tST22, tST23;
    double               tST33;

    // Rotate matrix T
    double tT11, tT12, tT13;
    double       tT22, tT23;
    double             tT33;

    // Eigenvalues
    double mu1, mu2, mu3, nu1, nu2, nu3;

    // Rotation matrix and normalization factors
    double O11, O12, O13;
    double O21, O22, O23;
    double O31, O32, O33;
    double m1, m2, m3;

    // Local copy of the Q Tensor (perturb to avoid degenerate eigenvector)
    double Q11 = Q[0][0][idx][0];
    double Q12 = Q[0][1][idx][0] + 1e-15;
    double Q13 = Q[0][2][idx][0] + 1e-15;
    double Q22 = Q[1][1][idx][0];
    double Q23 = Q[1][2][idx][0] + 1e-15;
    double Q33 = Q[2][2][idx][0];

    // Concentration
    double c = Q11 + Q22 + Q33; 

    // Normalize
    Q11 /= c;
    Q12 /= c;
    Q13 /= c;
    Q22 /= c;
    Q23 /= c;
    Q33 /= c;

    // Get gradient of velocity
    double E11 = gradU[0][0][idx][0]; 
    double E12 = 0.5 * (gradU[0][1][idx][0] + gradU[1][0][idx][0]); 
    double E13 = 0.5 * (gradU[0][2][idx][0] + gradU[2][0][idx][0]);
    double E22 = gradU[1][1][idx][0];
    double E23 = 0.5 * (gradU[1][2][idx][0] + gradU[2][1][idx][0]);
    double E33 = gradU[2][2][idx][0];

    // Construct and solve characteristic polynomial
    double mu = (1.0 / 3.0); //initial guess
    double a0 = Q11 * Q23 * Q23 + Q22 * Q13 * Q13 + Q33 * Q12 * Q12 - Q11 * Q22 * Q33 - 2 * Q13 * Q12 * Q23;
    double a1 = Q11 * Q22 + Q11 * Q33 + Q22 * Q33 - (Q12 * Q12 + Q13 * Q13 + Q23 * Q23);
   
    // Initial function evaluation
    double chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;
    
    // Start iteration count
    int iteration = 0;

    // Solve for one eigenvalue using Newton's method
    while(std::abs(chi_mu) > tolerance && iteration < maxIterations){

      // Evaluate function
      chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;

      // Compute update
      mu += -chi_mu / (3 * mu * mu - 2 * mu + a1);

      // Update iteration count
      iteration++;

    }

    // Get other eigenvalues
    nu1 = mu;
    nu2 = 0.5 * (-(mu - 1) + std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    nu3 = 0.5 * (-(mu - 1) - std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    
    // Sort
    mu1 = std::max(std::max(nu1, nu2), nu3);    
    mu3 = std::min(std::min(nu1, nu2), nu3);    
    mu2 = 1 - (mu1 + mu3);
    
    // First eigenvector
    O11 = Q12 * Q23 - Q13 * (Q22 - mu1);
    O21 = Q13 * Q12 - (Q11 - mu1) * Q23;
    O31 = (Q11 - mu1) * (Q22 - mu1) - Q12 * Q12;

    // Normalize
    m1 = std::sqrt(O11 * O11 + O21 * O21 + O31 * O31);
    O11 /= m1; 
    O21 /= m1; 
    O31 /= m1;

    // Second eigenvector
    O12 = Q12 * Q23 - Q13 * (Q22 - mu2);
    O22 = Q13 * Q12 - (Q11 - mu2) * Q23;
    O32 = (Q11 - mu2) * (Q22 - mu2) - Q12 * Q12;

    // Normalize
    m2 = std::sqrt(O12 * O12 + O22 * O22 + O32 * O32);
    O12 /= m2; 
    O22 /= m2; 
    O32 /= m2;

    // Third eigenvector
    O13 = O21 * O32 - O31 * O22;
    O23 = O31 * O12 - O11 * O32;
    O33 = O11 * O22 - O21 * O12;

    // Normalize
    m3 = std::sqrt(O13 * O13 + O23 * O23 + O33 * O33);
    O13 /= m3; 
    O23 /= m3; 
    O33 /= m3;

    // Improve orthogonality of second eigenvector via cross product
    O12 = O21 * O33 - O31 * O23;
    O22 = O31 * O13 - O11 * O33;
    O32 = O11 * O23 - O21 * O13;

    // Normalize
    m2 = std::sqrt(O12 * O12 + O22 * O22 + O32 * O32);
    O12 /= m2; 
    O22 /= m2; 
    O32 /= m2;

    // Evaluate Chebyshev interpolants

    // Domain transformation
    double pmu1 = (mu1 - 1.0 / 3.0) - (mu2 - 1.0 / 3.0);
    double pmu2 = 2 * (mu1 - 1.0 / 3.0) + 4 * (mu2 - 1.0 / 3.0);

    // Shift to [-1, 1]^2
    nu1 = 2 * std::min(std::max((pmu1 + pmu2), 0.0), 1.0) - 1; 
    nu2 = 2 * std::min(std::max(pmu1 / (pmu1 + pmu2), 0.0), 1.0) - 1;
    
    double Tim2 = 1.0, Tjm2 = 1.0;
    double Tim1 = nu1, Tjm1 = nu2;
    double Ti, Tj;

    // Evaluate coefficients (0, 0), (0, 1), (1, 0), (1, 1)
    tS1111 = Tim2 * (C11[0] * Tjm2 + C11[0 + maxChebDegree] * Tjm1) + Tim1 * (C11[1] * Tjm2 + C11[1 + maxChebDegree] * Tjm1);
    tS1122 = Tim2 * (C12[0] * Tjm2 + C12[0 + maxChebDegree] * Tjm1) + Tim1 * (C12[1] * Tjm2 + C12[1 + maxChebDegree] * Tjm1);
    tS2222 = Tim2 * (C22[0] * Tjm2 + C22[0 + maxChebDegree] * Tjm1) + Tim1 * (C22[1] * Tjm2 + C22[1 + maxChebDegree] * Tjm1);

    // Evaluate coefficients along (:, 0) and (:, 1)
    for(int ix = 2; ix < Ncheb; ix++){

      Ti = 2 * nu1 * Tim1 - Tim2;

      tS1111 += Ti * (C11[ix] * Tjm2 + C11[ix + maxChebDegree] * Tjm1);
      tS1122 += Ti * (C12[ix] * Tjm2 + C12[ix + maxChebDegree] * Tjm1);
      tS2222 += Ti * (C22[ix] * Tjm2 + C22[ix + maxChebDegree] * Tjm1);

      Tim2 = Tim1;
      Tim1 = Ti;

    }      

    // Evaluate via recursion relation
    for(int iy = 2; iy < Ncheb; iy++){

      Tj = 2 * nu2 * Tjm1 - Tjm2;

      Tim2 = 1.0;
      Tim1 = nu1;

      tS1111 += Tim2 * C11[0 + maxChebDegree * iy] * Tj + Tim1 * C11[1 + maxChebDegree * iy] * Tj;
      tS1122 += Tim2 * C12[0 + maxChebDegree * iy] * Tj + Tim1 * C12[1 + maxChebDegree * iy] * Tj;
      tS2222 += Tim2 * C22[0 + maxChebDegree * iy] * Tj + Tim1 * C22[1 + maxChebDegree * iy] * Tj;

      for(int ix = 2; ix < (Ncheb - iy); ix++){

        Ti = 2 * nu1 * Tim1 - Tim2;

        tS1111 += Ti * C11[ix + maxChebDegree * iy] * Tj;
        tS1122 += Ti * C12[ix + maxChebDegree * iy] * Tj;
        tS2222 += Ti * C22[ix + maxChebDegree * iy] * Tj;

        Tim2 = Tim1; 
        Tim1 = Ti;

      }

      Tjm2 = Tjm1; 
      Tjm1 = Tj;

    }

    // Evaluate remaining terms via trace identities
    tS1133 = mu1 - tS1111 - tS1122;
    tS2233 = mu2 - tS1122 - tS2222;
    tS3333 = mu3 - tS1133 - tS2233;

    // Compute rotations
    tT11 = O11 * (E11 * O11 + 2 * E12 * O21 + 2 * E13 * O31) +
           O21 * (E22 * O21 + 2 * E23 * O31) + 
           E33 * O31 * O31 + 2 * zeta * mu1;   
         
    tT12 = E11 * O11 * O12 + E12 * (O11 * O22 + O21 * O12) + 
           E22 * O21 * O22 + E13 * (O11 * O32 + O31 * O12) + 
           E33 * O31 * O32 + E23 * (O21 * O32 + O31 * O22);
         
    tT13 = E11 * O11 * O13 + E12 * (O11 * O23 + O21 * O13) + 
           E13 * (O11 * O33 + O31 * O13) + E22 * O21 * O23 + 
           E23 * (O21 * O33 + O31 * O23) + E33 * O31 * O33;
         
    tT22 = O12 * (E11 * O12 + 2 * E12 * O22 + 2 * E13 * O32) + 
           O22 * (E22 * O22 + 2 * E23 * O32) + 
           E33 * O32 * O32 + 2 * zeta * mu2;  
         
    tT23 = E11 * O12 * O13 + E12 * (O12 * O23 + O22 * O13) + 
           E13 * (O12 * O33 + O32 * O13) + E22 * O22 * O23 + 
           E23 * (O22 * O33 + O32 * O23) + E33 * O32 * O33;
         
    tT33 = O13 * (E11 * O13 + 2 * E12 * O23 + 2 * E13 * O33) + 
           O23 * (E22 * O23 + 2 * E23 * O33) + 
           E33 * O33 * O33 + 2 * zeta * mu3;  

         
    // Compute contraction in rotated frame
    tST11 = tS1111 * tT11 + tS1122 * tT22 + tS1133 * tT33;
    tST12 = 2 * tS1122 * tT12;
    tST13 = 2 * tS1133 * tT13;
    tST22 = tS1122 * tT11 + tS2222 * tT22 + tS2233 * tT33;
    tST23 = 2 * tS2233 * tT23;
    tST33 = tS1133 * tT11 + tS2233 * tT22 + tS3333 * tT33;

    // Rotate to physical frame and store
    ST[0][0][idx][0] = c * (O11 * (O11 * tST11 + O12 * tST12 + O13 * tST13) +
                          O12 * (O11 * tST12 + O12 * tST22 + O13 * tST23) + 
                          O13 * (O11 * tST13 + O12 * tST23 + O13 * tST33));

    ST[0][1][idx][0] = c * (O11 * (O21 * tST11 + O22 * tST12 + O23 * tST13) + 
                          O12 * (O21 * tST12 + O22 * tST22 + O23 * tST23) + 
                          O13 * (O21 * tST13 + O22 * tST23 + O23 * tST33));

    ST[0][2][idx][0] = c * (O11 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + 
                          O12 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + 
                          O13 * (O31 * tST13 + O32 * tST23 + O33 * tST33));

    ST[1][1][idx][0] = c * (O21 * (O21 * tST11 + O22 * tST12 + O23 * tST13) +
                          O22 * (O21 * tST12 + O22 * tST22 + O23 * tST23) +
                          O23 * (O21 * tST13 + O22 * tST23 + O23 * tST33));

    ST[1][2][idx][0] = c * (O21 * (O31 * tST11 + O32 * tST12 + O33 * tST13) +
                          O22 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + 
                          O23 * (O31 * tST13 + O32 * tST23 + O33 * tST33));

    ST[2][2][idx][0] = c * (O31 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + 
                          O32 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + 
                          O33 * (O31 * tST13 + O32 * tST23 + O33 * tST33));


  }
}

/************************************************************
  Stokes solver 

  This function solves for the velocity field via the Stokes equation

    -Delta(U) + grad(p) = div(Sigma)
                 div(U) = 0

  for an input stress tensor Sigma using a spectral discretization.

  Parameters:

    U (Vector) : velocity field, output
    U_h (Vector) : Fourier transform of the velocity field
    Sigma (Tensor) : stress tensor, input
    Sigma_h (Tensor) : Fourier transform of the stress tensor
    wavenumber (double*) : list of Fourier coefficients
    fft3, ifft3 (fftw_plan) : plans for forward and inverse Fourier transforms

************************************************************/
auto stokes(fftw_complex* u[3], fftw_complex* u_h[3], fftw_complex* Sigma[3][3], fftw_complex* Sigma_h[3][3], double* wavenumber, FastFourierTransform& fft){

  // Compute Fourier transform
  // fft(fft3, Sigma, Sigma_h);
  fft.fft(Sigma, Sigma_h);

  #pragma omp parallel for
  for(int ix = 0; ix < N; ix++){

    // x wavenumber
    double k1 = wavenumber[ix];

    for(int iy = 0; iy < N; iy++){
      
      // y wavenumber
      double k2 = wavenumber[iy];
      
      for(int iz = 0; iz < N; iz++){

        // z wavenumber
        double k3 = wavenumber[iz]; 

        // magnitude of wavevector
        double ksq = k1 * k1 + k2 * k2 + k3 * k3;

        // Flattened index
        auto idx = iz + N * iy + N * N * ix;  
        
        // Stokes operator
        double L11 = (1.0 / ksq) * (1.0 - k1 * k1 / ksq) * inv_Z;
        double L12 = (1.0 / ksq) * (0.0 - k1 * k2 / ksq) * inv_Z;
        double L13 = (1.0 / ksq) * (0.0 - k1 * k3 / ksq) * inv_Z;
        double L22 = (1.0 / ksq) * (1.0 - k2 * k2 / ksq) * inv_Z;
        double L23 = (1.0 / ksq) * (0.0 - k2 * k3 / ksq) * inv_Z;
        double L33 = (1.0 / ksq) * (1.0 - k3 * k3 / ksq) * inv_Z;

        fftw_complex rhs1, rhs2, rhs3;
       
        // Evaluate divergence
        rhs1[0] = -(k1 * Sigma_h[0][0][idx][1] + 
                    k2 * Sigma_h[0][1][idx][1] +
                    k3 * Sigma_h[0][2][idx][1]); 

        rhs2[0] = -(k1 * Sigma_h[1][0][idx][1] + 
                    k2 * Sigma_h[1][1][idx][1] +
                    k3 * Sigma_h[1][2][idx][1]); 
        
        rhs3[0] = -(k1 * Sigma_h[2][0][idx][1] + 
                    k2 * Sigma_h[2][1][idx][1] +
                    k3 * Sigma_h[2][2][idx][1]);  

        rhs1[1] = (k1 * Sigma_h[0][0][idx][0] + 
                   k2 * Sigma_h[0][1][idx][0] +
                   k3 * Sigma_h[0][2][idx][0]);

        rhs2[1] = (k1 * Sigma_h[1][0][idx][0] + 
                   k2 * Sigma_h[1][1][idx][0] +
                   k3 * Sigma_h[1][2][idx][0]); 

        rhs3[1] = (k1 * Sigma_h[2][0][idx][0] + 
                   k2 * Sigma_h[2][1][idx][0] +
                   k3 * Sigma_h[2][2][idx][0]);

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
  fft.ifft(u_h, u);

}

/************************************************************
  Fluid solver with fixed point iteration
************************************************************/
auto fluidSolver(fftw_complex* u[3], fftw_complex* u_h[3], fftw_complex* up1[3], fftw_complex* up1_h[3], fftw_complex* Q[3][3], fftw_complex* Q_h[3][3], fftw_complex* Sigma[3][3], fftw_complex* Sigma_h[3][3], fftw_complex* ST[3][3], fftw_complex* gradU[3][3],
  fftw_complex* buffer[3], double* wavenumber, double tolerance, int maxIterations, FastFourierTransform& fft){
  
    // Compute unconstrained stress Sigma = sigma_a * Q
    #pragma omp parallel for
    for(auto idx = 0; idx < N * N * N; idx++){

      Sigma[0][0][idx][0] = sigma_a * Q[0][0][idx][0];
      Sigma[0][1][idx][0] = sigma_a * Q[0][1][idx][0];
      Sigma[0][2][idx][0] = sigma_a * Q[0][2][idx][0];
      Sigma[1][1][idx][0] = sigma_a * Q[1][1][idx][0];
      Sigma[1][2][idx][0] = sigma_a * Q[1][2][idx][0];
      Sigma[2][2][idx][0] = sigma_a * Q[2][2][idx][0];

    }

    // Solve for velocity field with unconstrained stress
    stokes(u, u_h, Sigma, Sigma_h, wavenumber, fft);

    // If constraint stress is included, perform fixed iteration
    if(sigma_b != 0){

      // Set initial iteration counter
      int iteration = 0;

      // Set initial error
      double error = 10 * tolerance;
      
      while(error > tolerance && iteration < maxIterations){

        // Compute velocity gradient
        grad(u, u_h, gradU, buffer, wavenumber, fft);

        // Evaluate constraint term via Bingham closure
        binghamClosure(ST, Q, gradU);

        // Update stress
        stress(Sigma, Q, ST);

        // Solve for fluid velocity
        stokes(up1, up1_h, Sigma, Sigma_h, wavenumber, fft);

        // Initialize current iteration error
        error = 0;

        // Check update
        #pragma omp parallel for reduction(+:error)
        for(auto idx = 0; idx < N * N * N; idx++){
          error += std::abs(up1[0][idx][0] - u[0][idx][0]) +
                   std::abs(up1[1][idx][0] - u[1][idx][0]) +
                   std::abs(up1[2][idx][0] - u[2][idx][0]);
        }

        // Prepare for next iteration
        #pragma omp parallel for
        for(auto idx = 0; idx < N * N * N; idx++){
          u[0][idx][0] = up1[0][idx][0];
          u[1][idx][0] = up1[1][idx][0];
          u[2][idx][0] = up1[2][idx][0];
        }

        iteration++;

      }

    }

}

/************************************************************
  Scalar nematic order parameter
************************************************************/
auto nematicOrderParameter(fftw_complex* Q[3][3], double* s){
  
  double tolerance = 1e-14; //convergence tolerance for eig solve
  int maxIterations = 100; //std::max iterations for eig solve

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    
    double mu = (1.0 / 3.0); //initial guess
    double mup1, mu1, mu2, mu3, nu1, nu2, nu3;
    double O11, O21, O31, m1;

    // Local copy of the Q Tensor
    double Q11 = Q[0][0][idx][0];
    double Q12 = Q[0][1][idx][0];
    double Q13 = Q[0][2][idx][0];
    double Q22 = Q[1][1][idx][0];
    double Q23 = Q[1][2][idx][0];
    double Q33 = Q[2][2][idx][0];

    // Coefficients of characteristic polynomial
    double a0 = Q11 * Q23 * Q23 + Q22 * Q13 * Q13 + Q33 * Q12 * Q12 - Q11 * Q22 * Q33 - 2 * Q13 * Q12 * Q23;
    double a1 = Q11 * Q22 + Q11 * Q33 + Q22 * Q33 - (Q12 * Q12 + Q13 * Q13 + Q23 * Q23);

    // Initial function evaluation
    double chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;
 
    // Initialize iteration
    int iteration = 0; //iteration count

    // Find root of characteristic polynomial using Newtons method
    while(std::abs(chi_mu) > tolerance && iteration < maxIterations){

      // Evaluate characteristic polynomial
      chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;

      // Update
      mu -= chi_mu / (3 * mu * mu - 2 * mu + a1);

      iteration++;

    }

    // Get other eigenvalues
    nu1 = mu;
    nu2 = 0.5 * (-(mu - 1) + std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    nu3 = 0.5 * (-(mu - 1) - std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
    
    mu = std::max(std::max(nu1, nu2), nu3); //sort 
    s[idx] = 1.5 * (mu - 1.0 / 3); //store
    
  }
}

/************************************************************
  Enforce trace(Q) = 1
************************************************************/
auto retrace(fftw_complex* Q[3][3]){

  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    Q[2][2][idx][0] = 1.0 - Q[0][0][idx][0] - Q[1][1][idx][0];
  }

}

/************************************************************
  l2 norm of vector valued functions
************************************************************/
auto magnitude(fftw_complex* u[3], double* u_abs){

  // Loop over array
  #pragma omp parallel for
  for(auto idx = 0; idx < N * N * N; idx++){
    u_abs[idx] = std::sqrt(u[0][idx][0] * u[0][idx][0] + u[1][idx][0] * u[1][idx][0] + u[2][idx][0] * u[2][idx][0]);
  }

} 

/************************************************************
  L2 norm of vector valued functions over domain
************************************************************/
double norm_2(fftw_complex* u[3], double dV){

  // Initialize
  double u_norm = 0;

  // Loop over array
  #pragma omp parallel for reduction(+:u_norm)
  for(auto idx = 0; idx < N * N * N; idx++){
    for(auto i = 0; i < 3; i++){
      u_norm += u[i][idx][0] * u[i][idx][0];
    }
  }

  // Take square root and weight
  return std::sqrt(u_norm * dV);

} 

/************************************************************
  L infinity norm of vector valued functions over all components
************************************************************/
double norm_inf(fftw_complex* u[3]){

  // Initialize
  double u_norm = 0;

  // Loop over array
  #pragma omp parallel for reduction(max:u_norm)
  for(auto idx = 0; idx < N * N * N; idx++){
    for(auto i = 0; i < 3; i++){
      u_norm = std::max(u_norm, std::abs(u[i][idx][0]));
    }
  }

  // Return
  return u_norm;

} 

/************************************************************
  Plane wave perturbation
************************************************************/
auto planeWave(double* u, double* wavenumber){

  // Wavenumber
  double k1 = wavenumber[randi(4)];
  double k2 = wavenumber[randi(4)];
  double k3 = wavenumber[randi(4)];

  // Phase shift
  double w1 = 2 * pi * randf();
  double w2 = 2 * pi * randf();
  double w3 = 2 * pi * randf();

  // Perturbation magnitude
  double C = 0.001;

  // Fill in array
  for(auto ix = 0; ix < N; ix++){

    double x = (double) ix * (L / N);

    for(auto iy = 0; iy < N; iy++){

      double y = (double) iy * (L / N);

      for(auto iz = 0; iz < N; iz++){

        double z = (double) iz * (L / N);

        auto idx = iz + N * iy + N * N * ix;
        
        u[idx] = C * std::cos(k1 * x + w1) * std::cos(k2 * y + w2) * std::cos(k3 * z + w3);

      }
    }
  }

}

/************************************************************
  Evaluate sum of plane wave perturbations
************************************************************/
auto perturbation(fftw_complex* u, double* wavenumber){

  // Number of perturbations
  int Npert = 4;

  // Initialize
  double* du = new double[N * N * N];

  // Begin loop
  for(int n = 0; n < Npert; n++){

    // Construct perturbations
    planeWave(du, wavenumber);
  
    // Apply perturbation
    for(auto idx = 0; idx < N * N * N; idx++)
      u[idx][0] += du[idx];

  }

  // Clean up
  delete[] du;

}

/************************************************************
  Write field to file
************************************************************/
auto writeField(std::string filename, double* U, int precision){

  // Create file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(auto idx = 0; idx < N * N * N; idx++){
    wf << U[idx] << "\n";
  }

  wf.close();

}

auto writeField(std::string filename, fftw_complex* U, int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(auto idx = 0; idx < N * N * N; idx++){
    wf << U[idx][0] << "\n";
  }

  wf.close();

}

auto writeField(std::string filename, fftw_complex* u[3], int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(auto idx = 0; idx < N * N * N; idx++){
    wf << u[0][idx][0] << " " << u[1][idx][0] << " " << u[2][idx][0] << "\n";
  }
  
  wf.close();

}

auto writeField(std::string filename, fftw_complex* U[3][3], int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(auto idx = 0; idx < N * N * N; idx++){
    wf << U[0][0][idx][0] << " " << U[0][1][idx][0] << " " << U[0][2][idx][0] << " " 
       << U[1][1][idx][0] << " " << U[1][2][idx][0] << " " << U[2][2][idx][0] << "\n";
  }

  wf.close();

}

#endif
