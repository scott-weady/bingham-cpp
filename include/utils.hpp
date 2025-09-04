
#ifndef utils
#define utils


/* Creates a directory under "save" with name foldername */
void createFolder(std::string foldername){

  std::string message = "mkdir " + foldername;
  int check = system(message.c_str());

}

/************************************************************
  Random integer up to imax
************************************************************/
int randi(int imax){

  return (1 + (rand() % imax));

}

/************************************************************
  Random float between 0 and 1
************************************************************/
double randf(){

  return ((double) rand() / (RAND_MAX));

}

/************************************************************
  Vector class (rank 1)

  Properties:

    f1, f2, f3 (fftw_complex*): coordinate arrays
    num_elements (long): number of spatial discretization points

************************************************************/
class Vector {

public:

    // Create pointers for each element
    fftw_complex *f1, *f2, *f3;

    // Number of elements in each array
    long num_elements; 

    // Constructor
    Vector(long num_elements) : f1(nullptr), f2(nullptr), f3(nullptr) {

        // Calculate memory size for each fftw_complex array
        long mem_size = num_elements * sizeof(fftw_complex);

        // Allocate memory
        f1 = (fftw_complex*) malloc(mem_size);
        f2 = (fftw_complex*) malloc(mem_size);
        f3 = (fftw_complex*) malloc(mem_size);

        // Initialize memory to zero
        std::memset(f1, 0, mem_size);
        std::memset(f2, 0, mem_size);
        std::memset(f3, 0, mem_size);
    }

    // Method to free allocated memory
    void freeMemory() {

      free(f1);
      free(f2);
      free(f3);

    }

};

/************************************************************
  Tensor class (rank 2)

  Properties:
  
    f11, f12, f13, f21, f22, f23, f31, f32, f33 (fftw_complex*): coordinate arrays
    num_elements (long): number of spatial discretization points
    isSymmetric (bool): Boolean for symmetric matrix

************************************************************/
class Tensor {

public:

    // Create pointers for each element
    fftw_complex *f11, *f12, *f13;
    fftw_complex *f21, *f22, *f23;
    fftw_complex *f31, *f32, *f33;

    // Number of elements in each array
    long num_elements; 

    // Indicator for symmetric Tensor
    bool isSymmetric;

    // Constructor
    Tensor(long num_elements, bool isSymmetric) : 
            f11(nullptr), f12(nullptr), f13(nullptr),
            f21(nullptr), f22(nullptr), f23(nullptr),
            f31(nullptr), f32(nullptr), f33(nullptr),
            num_elements(num_elements), 
            isSymmetric(isSymmetric) {

        // Calculate memory size for each fftw_complex array
        long mem_size = num_elements * sizeof(fftw_complex);

        // Allocate memory
        f11 = (fftw_complex*) malloc(mem_size);
        f12 = (fftw_complex*) malloc(mem_size);
        f13 = (fftw_complex*) malloc(mem_size);

        f22 = (fftw_complex*) malloc(mem_size);
        f23 = (fftw_complex*) malloc(mem_size);

        f33 = (fftw_complex*) malloc(mem_size);

        // Initialize memory to zero
        std::memset(f11, 0, mem_size);
        std::memset(f12, 0, mem_size);
        std::memset(f13, 0, mem_size);

        std::memset(f22, 0, mem_size);
        std::memset(f23, 0, mem_size);

        std::memset(f33, 0, mem_size);

        // If symmetric, store symmetrized coordinate in same memory
        if(isSymmetric){
          
          f21 = f12;
          f31 = f13;
          f32 = f23;
        
        }
        else{

          f21 = (fftw_complex*) malloc(mem_size);
          f31 = (fftw_complex*) malloc(mem_size);
          f32 = (fftw_complex*) malloc(mem_size);

          std::memset(f21, 0, mem_size);
          std::memset(f31, 0, mem_size);
          std::memset(f32, 0, mem_size);

        }

    }

    // Method to free allocated memory
    void freeMemory() {

      free(f11);
      free(f12);
      free(f13);
      free(f22);
      free(f23);
      free(f33);

      if(!isSymmetric){
        free(f21);
        free(f31);
        free(f32);
      }

    }


};

/************************************************************
  Tensor class (rank 3)

  Assumed to be symmetric in first two components.

  Properties:

    fijk (fftw_complex*): coordinate arrays
    num_elements (long): number of spatial discretization points

************************************************************/
class Tensor3 {

public:

    // Create pointers for each element
    fftw_complex *f111, *f112, *f113;
    fftw_complex *f121, *f122, *f123;
    fftw_complex *f131, *f132, *f133;

    fftw_complex *f211, *f212, *f213;
    fftw_complex *f221, *f222, *f223;
    fftw_complex *f231, *f232, *f233;

    fftw_complex *f311, *f312, *f313;
    fftw_complex *f321, *f322, *f323;
    fftw_complex *f331, *f332, *f333;

    // Number of elements in each array
    long num_elements; 

    // Constructor
    Tensor3(long num_elements) : 
            f111(nullptr), f112(nullptr), f113(nullptr), 
            f121(nullptr), f122(nullptr), f123(nullptr),
            f131(nullptr), f132(nullptr), f133(nullptr),
            f211(nullptr), f212(nullptr), f213(nullptr),
            f221(nullptr), f222(nullptr), f223(nullptr),
            f231(nullptr), f232(nullptr), f233(nullptr),
            f311(nullptr), f312(nullptr), f313(nullptr),
            f321(nullptr), f322(nullptr), f323(nullptr),
            f331(nullptr), f332(nullptr), f333(nullptr),
            num_elements(num_elements) {

        // Calculate memory size for each fftw_complex array
        long mem_size = num_elements * sizeof(fftw_complex);

        // Allocate memory
        f111 = (fftw_complex*) malloc(mem_size);
        f112 = (fftw_complex*) malloc(mem_size);
        f113 = (fftw_complex*) malloc(mem_size);

        f121 = (fftw_complex*) malloc(mem_size);
        f122 = (fftw_complex*) malloc(mem_size);
        f123 = (fftw_complex*) malloc(mem_size);
        
        f131 = (fftw_complex*) malloc(mem_size);
        f132 = (fftw_complex*) malloc(mem_size);
        f133 = (fftw_complex*) malloc(mem_size);
        
        f211 = f121;
        f212 = f122;
        f213 = f123;

        f221 = (fftw_complex*) malloc(mem_size);
        f222 = (fftw_complex*) malloc(mem_size);
        f223 = (fftw_complex*) malloc(mem_size);
        
        f231 = (fftw_complex*) malloc(mem_size);
        f232 = (fftw_complex*) malloc(mem_size);
        f233 = (fftw_complex*) malloc(mem_size);
        
        f311 = f131;
        f312 = f132;
        f313 = f133;

        f321 = f231;
        f322 = f232;
        f323 = f233;

        f331 = (fftw_complex*) malloc(mem_size);
        f332 = (fftw_complex*) malloc(mem_size);
        f333 = (fftw_complex*) malloc(mem_size);

        // Initialize memory to zero
        std::memset(f111, 0, mem_size);
        std::memset(f112, 0, mem_size);
        std::memset(f113, 0, mem_size);

        std::memset(f121, 0, mem_size);
        std::memset(f122, 0, mem_size);
        std::memset(f123, 0, mem_size);

        std::memset(f131, 0, mem_size);
        std::memset(f132, 0, mem_size);
        std::memset(f133, 0, mem_size);

        std::memset(f221, 0, mem_size);
        std::memset(f222, 0, mem_size);
        std::memset(f223, 0, mem_size);

        std::memset(f231, 0, mem_size);
        std::memset(f232, 0, mem_size);
        std::memset(f233, 0, mem_size);

        std::memset(f331, 0, mem_size);
        std::memset(f332, 0, mem_size);
        std::memset(f333, 0, mem_size);

    }

    // Free allocated memory
    void freeMemory() {

      free(f111);
      free(f112);
      free(f113);

      free(f121);
      free(f122);
      free(f123);

      free(f131);
      free(f132);
      free(f133);

      free(f221);
      free(f222);
      free(f223);

      free(f231);
      free(f232);
      free(f233);

      free(f331);
      free(f332);
      free(f333);

    }

};


class FastFourierTransform {

  public:
    
    fftw_plan fft3_plan, ifft3_plan;
    fftw_complex *in, *out;

    // Constructor
    FastFourierTransform(int N) {

      // Initialize arrays for FFT planner
      in = (fftw_complex*) malloc(N * N * N * sizeof(fftw_complex));
      out = (fftw_complex*) malloc(N * N * N * sizeof(fftw_complex)); 

      // Forward transform
      fft3_plan = fftw_plan_dft_3d(N, N, N, in, out, -1, FFTW_MEASURE); 

      // Inverse transform
      ifft3_plan = fftw_plan_dft_3d(N, N, N, in, out, 1, FFTW_MEASURE);

    }

    ~FastFourierTransform() {
        fftw_destroy_plan(fft3_plan);
        fftw_destroy_plan(ifft3_plan);
        fftw_free(in);
        fftw_free(out);
    }

    // Scalar (rank 0)
    void fft(fftw_complex* u, fftw_complex* u_h){
      fftw_execute_dft(fft3_plan, u, u_h);
    }

    // Vector (rank 1)
    void fft(Vector u, Vector u_h){  
      fft(u.f1, u_h.f1);
      fft(u.f2, u_h.f2);
      fft(u.f3, u_h.f3);
    }

    // Tensor (rank 2)
    void fft(Tensor u, Tensor u_h){

      fft(u.f11, u_h.f11);
      fft(u.f12, u_h.f12);
      fft(u.f13, u_h.f13);
      fft(u.f22, u_h.f22);
      fft(u.f23, u_h.f23);
      fft(u.f33, u_h.f33);

      if(!u.isSymmetric){

        fft(u.f21, u_h.f21);
        fft(u.f31, u_h.f31);
        fft(u.f32, u_h.f32);
        
      }

    }

    void fft(fftw_complex* u[3], fftw_complex* u_h[3]){
      for(int i = 0; i < 3; i++){
        fft(u[i], u_h[i]);
      }
    }

    void fft(fftw_complex* u[3][3], fftw_complex* u_h[3][3]){
      for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
          fft(u[i][j], u_h[i][j]);
        }
      }
    }

    // Inverse transforms

    // Scalar (rank 0)
    void ifft(fftw_complex* u_h, fftw_complex* u){
      fftw_execute_dft(ifft3_plan, u_h, u);
    }

    // Vector (rank 1)
    void ifft(Vector u_h, Vector u){  
      ifft(u_h.f1, u.f1);
      ifft(u_h.f2, u.f2);
      ifft(u_h.f3, u.f3);
    }

    // Tensor (rank 2)
    void ifft(Tensor u_h, Tensor u){

      ifft(u_h.f11, u.f11);
      ifft(u_h.f12, u.f12);
      ifft(u_h.f13, u.f13);
      ifft(u_h.f22, u.f22);
      ifft(u_h.f23, u.f23);
      ifft(u_h.f33, u.f33);
      
      if(!u_h.isSymmetric){
        ifft(u_h.f21, u.f21);
        ifft(u_h.f31, u.f31);
        ifft(u_h.f32, u.f32);
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
void helmholtzOperator(double* Operator, double* wavenumber, double c){

  // Fill in operator
  #pragma omp parallel for
  for(int ix = 0; ix < N; ix++){
    for(int iy = 0; iy < N; iy++){
      for(int iz = 0; iz < N; iz++){

        // Flattened index
        long idx = iz + N * iy + N * N * ix;

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
void grad(fftw_complex* u_h, Vector gradU_h, double* wavenumber){

  // Loop over all points
  #pragma omp parallel for
  for(int ix = 0; ix < N; ix++) {
    
    double k1 = wavenumber[ix];
    
    for(int iy = 0; iy < N; iy++) {
      
      double k2 = wavenumber[iy];
      
      for(int iz = 0; iz < N; iz++) {
      
        double k3 = wavenumber[iz]; 
        
        // Compute flattened index for 3D array
        long idx = iz + N * iy + N * N * ix;
        
        // Get real and imaginary parts
        double u_h_real = u_h[idx][0];
        double u_h_imag = u_h[idx][1];

        // Compute derivatives in each direction
        gradU_h.f1[idx][0] = -k1 * u_h_imag * inv_Z; // x (real part)
        gradU_h.f1[idx][1] =  k1 * u_h_real * inv_Z; // x (imaginary part)
        
        gradU_h.f2[idx][0] = -k2 * u_h_imag * inv_Z; // y (real part)
        gradU_h.f2[idx][1] =  k2 * u_h_real * inv_Z; // y (imaginary part)
        
        gradU_h.f3[idx][0] = -k3 * u_h_imag * inv_Z; // z (real part)
        gradU_h.f3[idx][1] =  k3 * u_h_real * inv_Z; // z (imaginary part)
        
      }
    } 
  }
}

// Vector, convention is grad(U)_ij = dU_i / dx_j
void grad(Vector U, Vector U_h, Tensor gradU, Vector bufferVector, double* wavenumber, FastFourierTransform& fft){

  // Compute Fourier transform
  fft.fft(U, U_h);

  // Compute the gradient of each component
  grad(U_h.f1, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f11);
  fft.ifft(bufferVector.f2, gradU.f12);
  fft.ifft(bufferVector.f3, gradU.f13);

  grad(U_h.f2, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f21);
  fft.ifft(bufferVector.f2, gradU.f22);
  fft.ifft(bufferVector.f3, gradU.f23);
  
  grad(U_h.f3, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f31);
  fft.ifft(bufferVector.f2, gradU.f32);
  fft.ifft(bufferVector.f3, gradU.f33);

}

// Rank-2 tensor (assumed symmetric), convention is grad(U)_ijk = dU_ij / dx_k
void grad(Tensor U, Tensor U_h, Tensor3 gradU, Vector bufferVector, double* wavenumber, FastFourierTransform& fft){

  // Compute Fourier transform
  fft.fft(U, U_h);

  // Compute the gradient of each component
  grad(U_h.f11, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f111);
  fft.ifft(bufferVector.f2, gradU.f112);
  fft.ifft(bufferVector.f3, gradU.f113);

  grad(U_h.f12, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f121);
  fft.ifft(bufferVector.f2, gradU.f122);
  fft.ifft(bufferVector.f3, gradU.f123);

  grad(U_h.f13, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f131);
  fft.ifft(bufferVector.f2, gradU.f132);
  fft.ifft(bufferVector.f3, gradU.f133);
  
  grad(U_h.f22, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f221);
  fft.ifft(bufferVector.f2, gradU.f222);
  fft.ifft(bufferVector.f3, gradU.f223);

  grad(U_h.f23, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f231);
  fft.ifft(bufferVector.f2, gradU.f232);
  fft.ifft(bufferVector.f3, gradU.f233);
  
  grad(U_h.f33, bufferVector, wavenumber);
  fft.ifft(bufferVector.f1, gradU.f331);
  fft.ifft(bufferVector.f2, gradU.f332);
  fft.ifft(bufferVector.f3, gradU.f333);

}

/************************************************************
  Anti-aliasing via the 2 / 3 rule
  
  Parameters:

    U_h (Tensor) : field in Fourier space, overwritten
    wavenumber (double*) : N x 1 array of Fourier modes
    kmax (double) : maximum wave number

************************************************************/
void antialias(Tensor U_h, double* wavenumber, double kmax){

  // Loop over all points
  #pragma omp parallel for
  for(long ix = 0; ix < N; ix++){

    double k1 = wavenumber[ix];

    for(long iy = 0; iy < N; iy++){

      double k2 = wavenumber[iy];

      for(long iz = 0; iz < N; iz++){

        double k3 = wavenumber[iz];

        if(std::abs(k1) > (2.0 / 3.0) * kmax || 
           std::abs(k2) > (2.0 / 3.0) * kmax || 
           std::abs(k3) > (2.0 / 3.0) * kmax){

          // Flattened index
          long idx = iz + N * iy + N * N * ix;

          U_h.f11[idx][0] = 0;
          U_h.f12[idx][0] = 0;
          U_h.f13[idx][0] = 0;
          U_h.f22[idx][0] = 0;
          U_h.f23[idx][0] = 0;
          U_h.f33[idx][0] = 0;

          U_h.f11[idx][1] = 0;
          U_h.f12[idx][1] = 0;
          U_h.f13[idx][1] = 0;
          U_h.f22[idx][1] = 0;
          U_h.f23[idx][1] = 0;
          U_h.f33[idx][1] = 0;

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
void euler(Tensor U, Tensor F, double* inverseLinearOperator, double dt){

  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){

    double Linv = inverseLinearOperator[idx]; 

    for(int k = 0; k < 2; k++){

      // Update, overwriting the current tensor
      U.f11[idx][k] = Linv * euler(U.f11[idx][k], F.f11[idx][k], dt);
      U.f12[idx][k] = Linv * euler(U.f12[idx][k], F.f12[idx][k], dt);
      U.f13[idx][k] = Linv * euler(U.f13[idx][k], F.f13[idx][k], dt);
      U.f22[idx][k] = Linv * euler(U.f22[idx][k], F.f22[idx][k], dt);
      U.f23[idx][k] = Linv * euler(U.f23[idx][k], F.f23[idx][k], dt);
      U.f33[idx][k] = Linv * euler(U.f33[idx][k], F.f33[idx][k], dt);

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
// double sbdf2(double U, double Um1, double F, double Fm1, double dt){

//   return (4.0 * U - Um1) / 3.0 + (2.0 * dt / 3.0) * (2.0 * F - Fm1);

// }

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
void sbdf2(Tensor U, Tensor Um1, Tensor F, Tensor Fm1, double* inverseLinearOperator, double dt, double dtm1){

  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){

    double Linv = inverseLinearOperator[idx]; 

    for(int k = 0; k < 2; k++){

      // Get solution at current time
      double u11 = U.f11[idx][k], u12 = U.f12[idx][k], u13 = U.f13[idx][k];
      double                      u22 = U.f22[idx][k], u23 = U.f23[idx][k];
      double                                           u33 = U.f33[idx][k];

      double f11 = F.f11[idx][k], f12 = F.f12[idx][k], f13 = F.f13[idx][k];
      double                      f22 = F.f22[idx][k], f23 = F.f23[idx][k];
      double                                           f33 = F.f33[idx][k];

      // Compute solution at next time
      U.f11[idx][k] = Linv * sbdf2(u11, Um1.f11[idx][k], f11, Fm1.f11[idx][k], dt, dtm1);
      U.f12[idx][k] = Linv * sbdf2(u12, Um1.f12[idx][k], f12, Fm1.f12[idx][k], dt, dtm1);
      U.f13[idx][k] = Linv * sbdf2(u13, Um1.f13[idx][k], f13, Fm1.f13[idx][k], dt, dtm1);
      U.f22[idx][k] = Linv * sbdf2(u22, Um1.f22[idx][k], f22, Fm1.f22[idx][k], dt, dtm1);
      U.f23[idx][k] = Linv * sbdf2(u23, Um1.f23[idx][k], f23, Fm1.f23[idx][k], dt, dtm1);
      U.f33[idx][k] = Linv * sbdf2(u33, Um1.f33[idx][k], f33, Fm1.f33[idx][k], dt, dtm1);

      // Overwrite previous values
      Um1.f11[idx][k] = u11;  
      Um1.f12[idx][k] = u12;  
      Um1.f13[idx][k] = u13;
      Um1.f22[idx][k] = u22;  
      Um1.f23[idx][k] = u23;  
      Um1.f33[idx][k] = u33;

      Fm1.f11[idx][k] = f11;  
      Fm1.f12[idx][k] = f12;
      Fm1.f13[idx][k] = f13;
      Fm1.f22[idx][k] = f22;
      Fm1.f23[idx][k] = f23;
      Fm1.f33[idx][k] = f33;

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
void nonlinear(Tensor F, Vector U, Tensor gradU, Tensor Q, Tensor3 gradQ, Tensor ST){

  // Main loop
  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
   
    // Load Q tensor
    double Q11 = Q.f11[idx][0], Q12 = Q.f12[idx][0], Q13 = Q.f13[idx][0];
    double Q21 = Q12,           Q22 = Q.f22[idx][0], Q23 = Q.f23[idx][0];
    double Q31 = Q13,           Q32 = Q23,           Q33 = Q.f33[idx][0]; 
    
    // Get S : (E + 2 * zeta * Q)
    double ST11 = ST.f11[idx][0], ST12 = ST.f12[idx][0], ST13 = ST.f13[idx][0];
    double                        ST22 = ST.f22[idx][0], ST23 = ST.f23[idx][0];
    double                                               ST33 = ST.f33[idx][0];
    
    // Load velocity
    double u = U.f1[idx][0];
    double v = U.f2[idx][0];
    double w = U.f3[idx][0];

    // Load velocity gradient
    double ux = gradU.f11[idx][0], uy = gradU.f12[idx][0], uz = gradU.f13[idx][0]; 
    double vx = gradU.f21[idx][0], vy = gradU.f22[idx][0], vz = gradU.f23[idx][0]; 
    double wx = gradU.f31[idx][0], wy = gradU.f32[idx][0], wz = gradU.f33[idx][0]; 
    
    // Compute upper convected derivative
    double Qp11 = u * gradQ.f111[idx][0] + v * gradQ.f112[idx][0] + w * gradQ.f113[idx][0] - 2 * (ux * Q11 + uy * Q21 + uz * Q31); 
    double Qp12 = u * gradQ.f121[idx][0] + v * gradQ.f122[idx][0] + w * gradQ.f123[idx][0] - (ux * Q12 + uy * Q22 + uz * Q32 + vx * Q11 + vy * Q12 + vz * Q13);
    double Qp13 = u * gradQ.f131[idx][0] + v * gradQ.f132[idx][0] + w * gradQ.f133[idx][0] - (ux * Q13 + uy * Q23 + uz * Q33 + wx * Q11 + wy * Q12 + wz * Q13); 
    double Qp22 = u * gradQ.f221[idx][0] + v * gradQ.f222[idx][0] + w * gradQ.f223[idx][0] - 2 *(vx * Q12 + vy * Q22 + vz * Q32); 
    double Qp23 = u * gradQ.f231[idx][0] + v * gradQ.f232[idx][0] + w * gradQ.f233[idx][0] - (vx * Q13 + vy * Q23 + vz * Q33 + wx * Q21 + wy * Q22 + wz * Q23); 
    double Qp33 = u * gradQ.f331[idx][0] + v * gradQ.f332[idx][0] + w * gradQ.f333[idx][0] - 2 * (wx * Q13 + wy * Q23 + wz * Q33);

    // Write nonlinear term
    F.f11[idx][0] = -Qp11 - 2 * ST11 + 4 * zeta * (Q11 * Q11 + Q12 * Q21 + Q13 * Q31) - 6 * dR * (Q11 - 1.0/3.0);
    F.f12[idx][0] = -Qp12 - 2 * ST12 + 4 * zeta * (Q11 * Q12 + Q12 * Q22 + Q13 * Q32) - 6 * dR * Q12;
    F.f13[idx][0] = -Qp13 - 2 * ST13 + 4 * zeta * (Q11 * Q13 + Q12 * Q23 + Q13 * Q33) - 6 * dR * Q13;
    F.f22[idx][0] = -Qp22 - 2 * ST22 + 4 * zeta * (Q21 * Q12 + Q22 * Q22 + Q23 * Q32) - 6 * dR * (Q22 - 1.0/3.0);
    F.f23[idx][0] = -Qp23 - 2 * ST23 + 4 * zeta * (Q21 * Q13 + Q22 * Q23 + Q23 * Q33) - 6 * dR * Q23;
    F.f33[idx][0] = -Qp33 - 2 * ST33 + 4 * zeta * (Q31 * Q13 + Q32 * Q23 + Q33 * Q33) - 6 * dR * (Q33 - 1.0/3.0);

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
void stress(Tensor Sigma, Tensor Q, Tensor ST){

  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
     
    // Load Q tensor
    double Q11 = Q.f11[idx][0], Q12 = Q.f12[idx][0], Q13 = Q.f13[idx][0];
    double Q21 = Q12,           Q22 = Q.f22[idx][0], Q23 = Q.f23[idx][0];
    double Q31 = Q13,           Q32 = Q23,           Q33 = Q.f33[idx][0]; 

    // Load constraint stress
    double ST11 = ST.f11[idx][0], ST12 = ST.f12[idx][0], ST13 = ST.f13[idx][0];
    double                        ST22 = ST.f22[idx][0], ST23 = ST.f23[idx][0];
    double                                               ST33 = ST.f33[idx][0];

    // Compute Q * Q
    double QQ11 = Q11 * Q11 + Q12 * Q21 + Q13 * Q31;
    double QQ12 = Q11 * Q12 + Q12 * Q22 + Q13 * Q32;
    double QQ13 = Q11 * Q13 + Q12 * Q23 + Q13 * Q33;
    double QQ22 = Q21 * Q12 + Q22 * Q22 + Q23 * Q32;
    double QQ23 = Q21 * Q13 + Q22 * Q23 + Q23 * Q33;
    double QQ33 = Q31 * Q13 + Q32 * Q23 + Q33 * Q33;
    
    // Write stress tensor
    Sigma.f11[idx][0] = sigma_a * Q11 + sigma_b * ST11 - 2 * sigma_b * zeta * QQ11;
    Sigma.f12[idx][0] = sigma_a * Q12 + sigma_b * ST12 - 2 * sigma_b * zeta * QQ12;
    Sigma.f13[idx][0] = sigma_a * Q13 + sigma_b * ST13 - 2 * sigma_b * zeta * QQ13;
    Sigma.f22[idx][0] = sigma_a * Q22 + sigma_b * ST22 - 2 * sigma_b * zeta * QQ22;
    Sigma.f23[idx][0] = sigma_a * Q23 + sigma_b * ST23 - 2 * sigma_b * zeta * QQ23;
    Sigma.f33[idx][0] = sigma_a * Q33 + sigma_b * ST33 - 2 * sigma_b * zeta * QQ33;

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
void binghamClosure(Tensor ST, Tensor Q, Tensor gradU){
  
  // Tolerance for eigenvalue solve
  double tolerance = 1e-14;

  // Max iterations for eigenvalue solve 
  int maxIterations = 50; 

  // Loop over all grid points
  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
    
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
    double Q11 = Q.f11[idx][0];
    double Q12 = Q.f12[idx][0] + 1e-15;
    double Q13 = Q.f13[idx][0] + 1e-15;
    double Q22 = Q.f22[idx][0];
    double Q23 = Q.f23[idx][0] + 1e-15;
    double Q33 = Q.f33[idx][0]; 

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
    double E11 = gradU.f11[idx][0]; 
    double E12 = 0.5 * (gradU.f12[idx][0] + gradU.f21[idx][0]); 
    double E13 = 0.5 * (gradU.f13[idx][0] + gradU.f31[idx][0]);
    double E22 = gradU.f22[idx][0];
    double E23 = 0.5 * (gradU.f23[idx][0] + gradU.f32[idx][0]);
    double E33 = gradU.f33[idx][0];

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
    ST.f11[idx][0] = c * (O11 * (O11 * tST11 + O12 * tST12 + O13 * tST13) +
                          O12 * (O11 * tST12 + O12 * tST22 + O13 * tST23) + 
                          O13 * (O11 * tST13 + O12 * tST23 + O13 * tST33));

    ST.f12[idx][0] = c * (O11 * (O21 * tST11 + O22 * tST12 + O23 * tST13) + 
                          O12 * (O21 * tST12 + O22 * tST22 + O23 * tST23) + 
                          O13 * (O21 * tST13 + O22 * tST23 + O23 * tST33));

    ST.f13[idx][0] = c * (O11 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + 
                          O12 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + 
                          O13 * (O31 * tST13 + O32 * tST23 + O33 * tST33));

    ST.f22[idx][0] = c * (O21 * (O21 * tST11 + O22 * tST12 + O23 * tST13) +
                          O22 * (O21 * tST12 + O22 * tST22 + O23 * tST23) +
                          O23 * (O21 * tST13 + O22 * tST23 + O23 * tST33));

    ST.f23[idx][0] = c * (O21 * (O31 * tST11 + O32 * tST12 + O33 * tST13) +
                          O22 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + 
                          O23 * (O31 * tST13 + O32 * tST23 + O33 * tST33));

    ST.f33[idx][0] = c * (O31 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + 
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
void stokes(Vector U, Vector U_h, Tensor Sigma, Tensor Sigma_h, double* wavenumber, FastFourierTransform& fft){
  
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
        long idx = iz + N * iy + N * N * ix;  
        
        // Stokes operator
        double L11 = (1.0 / ksq) * (1.0 - k1 * k1 / ksq) * inv_Z;
        double L12 = (1.0 / ksq) * (0.0 - k1 * k2 / ksq) * inv_Z;
        double L13 = (1.0 / ksq) * (0.0 - k1 * k3 / ksq) * inv_Z;
        double L22 = (1.0 / ksq) * (1.0 - k2 * k2 / ksq) * inv_Z;
        double L23 = (1.0 / ksq) * (0.0 - k2 * k3 / ksq) * inv_Z;
        double L33 = (1.0 / ksq) * (1.0 - k3 * k3 / ksq) * inv_Z;

        fftw_complex rhs1, rhs2, rhs3;
       
        // Evaluate divergence
        rhs1[0] = -(k1 * Sigma_h.f11[idx][1] + 
                    k2 * Sigma_h.f12[idx][1] +
                    k3 * Sigma_h.f13[idx][1]); 

        rhs2[0] = -(k1 * Sigma_h.f12[idx][1] + 
                    k2 * Sigma_h.f22[idx][1] +
                    k3 * Sigma_h.f23[idx][1]); 
        
        rhs3[0] = -(k1 * Sigma_h.f13[idx][1] + 
                    k2 * Sigma_h.f23[idx][1] +
                    k3 * Sigma_h.f33[idx][1]);  

        rhs1[1] = (k1 * Sigma_h.f11[idx][0] + 
                   k2 * Sigma_h.f12[idx][0] +
                   k3 * Sigma_h.f13[idx][0]);

        rhs2[1] = (k1 * Sigma_h.f12[idx][0] + 
                   k2 * Sigma_h.f22[idx][0] +
                   k3 * Sigma_h.f23[idx][0]); 

        rhs3[1] = (k1 * Sigma_h.f13[idx][0] + 
                   k2 * Sigma_h.f23[idx][0] +
                   k3 * Sigma_h.f33[idx][0]);

        for(int k = 0; k < 2; k++){ 

          U_h.f1[idx][k] = L11 * rhs1[k] + L12 * rhs2[k] + L13 * rhs3[k];
          U_h.f2[idx][k] = L12 * rhs1[k] + L22 * rhs2[k] + L23 * rhs3[k];
          U_h.f3[idx][k] = L13 * rhs1[k] + L23 * rhs2[k] + L33 * rhs3[k];
        
        }

      }
    }
  }

  // Set int(U) = 0
  U_h.f1[0][0] = 0.0;
  U_h.f1[0][1] = 0.0;

  U_h.f2[0][0] = 0.0;
  U_h.f2[0][1] = 0.0;

  U_h.f3[0][0] = 0.0;
  U_h.f3[0][1] = 0.0;
 
  // Convert to real space
  // fft(ifft3,U_h,U);
  fft.ifft(U_h, U);

}

/************************************************************
  Fluid solver with fixed point iteration
************************************************************/
void fluidSolver(Vector U, Vector U_h, Vector Up1, Vector Up1_h, Tensor Q, Tensor Q_h, Tensor Sigma, Tensor Sigma_h, Tensor ST, Tensor gradU,
  Vector bufferVector, double* wavenumber, double tolerance, int maxIterations, FastFourierTransform& fft){
  
    // Compute unconstrained stress Sigma = sigma_a * Q
    #pragma omp parallel for
    for(long idx = 0; idx < N * N * N; idx++){

      Sigma.f11[idx][0] = sigma_a * Q.f11[idx][0];
      Sigma.f12[idx][0] = sigma_a * Q.f12[idx][0];
      Sigma.f13[idx][0] = sigma_a * Q.f13[idx][0];
      Sigma.f22[idx][0] = sigma_a * Q.f22[idx][0];
      Sigma.f23[idx][0] = sigma_a * Q.f23[idx][0];
      Sigma.f33[idx][0] = sigma_a * Q.f33[idx][0];

    }

    // Solve for velocity field with unconstrained stress
    stokes(U, U_h, Sigma, Sigma_h, wavenumber, fft);

    // If constraint stress is included, perform fixed iteration
    if(sigma_b != 0){

      // Set initial iteration counter
      int iteration = 0;

      // Set initial error
      double error = 10 * tolerance;
      
      while(error > tolerance && iteration < maxIterations){

        // Compute velocity gradient
        grad(U, U_h, gradU, bufferVector, wavenumber, fft);

        // Evaluate constraint term via Bingham closure
        binghamClosure(ST, Q, gradU);

        // Update stress
        stress(Sigma, Q, ST);

        // Solve for fluid velocity
        stokes(Up1, Up1_h, Sigma, Sigma_h, wavenumber, fft);

        // Initialize current iteration error
        error = 0;

        // Check update
        #pragma omp parallel for reduction(+:error)
        for(long idx = 0; idx < N * N * N; idx++){
          error += std::abs(Up1.f1[idx][0] - U.f1[idx][0]) +
                   std::abs(Up1.f2[idx][0] - U.f2[idx][0]) +
                   std::abs(Up1.f3[idx][0] - U.f3[idx][0]);
        }

        // Prepare for next iteration
        #pragma omp parallel for
        for(long idx = 0; idx < N * N * N; idx++){
          U.f1[idx][0] = Up1.f1[idx][0];
          U.f2[idx][0] = Up1.f2[idx][0];
          U.f3[idx][0] = Up1.f3[idx][0];
        }

        iteration++;

      }

    }

}

/************************************************************
  Scalar nematic order parameter
************************************************************/
void nematicOrderParameter(Tensor Q, double* s){
  
  double tolerance = 1e-14; //convergence tolerance for eig solve
  int maxIterations = 100; //std::max iterations for eig solve

  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
    
    double mu = (1.0 / 3.0); //initial guess
    double mup1, mu1, mu2, mu3, nu1, nu2, nu3;
    double O11, O21, O31, m1;

    // Local copy of the Q Tensor
    double Q11 = Q.f11[idx][0];
    double Q12 = Q.f12[idx][0];
    double Q13 = Q.f13[idx][0];
    double Q22 = Q.f22[idx][0];
    double Q23 = Q.f23[idx][0];
    double Q33 = Q.f33[idx][0];

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
void retrace(Tensor Q){

  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
    Q.f33[idx][0] = 1 - Q.f11[idx][0] - Q.f22[idx][0];
  }

}

/************************************************************
  l2 norm of vector valued functions
************************************************************/
void magnitude(Vector U, double* U_abs){

  // Loop over array
  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
    U_abs[idx] = std::sqrt(U.f1[idx][0] * U.f1[idx][0] + U.f2[idx][0] * U.f2[idx][0] + U.f3[idx][0] * U.f3[idx][0]);
  }

} 

/************************************************************
  L2 norm of vector valued functions over domain
************************************************************/
double norm_2(Vector U, double dV){

  // Initialize
  double U_norm = 0;

  // Loop over array
  #pragma omp parallel for reduction(+:U_norm)
  for(long idx = 0; idx < N * N * N; idx++){
    U_norm += U.f1[idx][0] * U.f1[idx][0] + U.f2[idx][0] * U.f2[idx][0] + U.f3[idx][0] * U.f3[idx][0];
  }

  // Take square root and weight
  U_norm = std::sqrt(U_norm * dV);

  // Return
  return U_norm;

} 

/************************************************************
  L infinity norm of vector valued functions over all components
************************************************************/
double norm_inf(Vector U){

  // Initialize
  double U_norm = 0;

  // Loop over array
  #pragma omp parallel for
  for(long idx = 0; idx < N * N * N; idx++){
    U_norm = std::max(U_norm, std::abs(U.f1[idx][0]));
    U_norm = std::max(U_norm, std::abs(U.f2[idx][0]));
    U_norm = std::max(U_norm, std::abs(U.f3[idx][0]));
  }

  // Return
  return U_norm;

} 

/************************************************************
  Plane wave perturbation
************************************************************/
void planeWave(double* u, double* wavenumber){

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
  for(long ix = 0; ix < N; ix++){

    double x = (double) ix * (L / N);

    for(long iy = 0; iy < N; iy++){

      double y = (double) iy * (L / N);

      for(long iz = 0; iz < N; iz++){

        double z = (double) iz * (L / N);

        long idx = iz + N * iy + N * N * ix;
        
        u[idx] = C * std::cos(k1 * x + w1) * std::cos(k2 * y + w2) * std::cos(k3 * z + w3);

      }
    }
  }

}

/************************************************************
  Evaluate sum of plane wave perturbations
************************************************************/
void perturbation(fftw_complex* u, double* wavenumber){

  // Number of perturbations
  int Npert = 4;

  // Initialize
  double* du = (double*) malloc(N * N * N * sizeof(double));

  // Begin loop
  for(int n = 0; n < Npert; n++){

    // Construct perturbations
    planeWave(du, wavenumber);
  
    // Apply perturbation
    for(long idx = 0; idx < N * N * N; idx++)
      u[idx][0] += du[idx];

  }

  // Clean up
  free(du);

}

/************************************************************
  Write field to file
************************************************************/
void writeField(std::string filename, double* U, int precision){

  // Create file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(long idx = 0; idx < N * N * N; idx++){
    wf << U[idx] << "\n";
  }

  wf.close();

}

void writeField(std::string filename, fftw_complex* U, int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(long idx = 0; idx < N * N * N; idx++){
    wf << U[idx][0] << "\n";
  }

  wf.close();

}


void writeField(std::string filename, Vector U, int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(long idx = 0; idx < N * N * N; idx++){
    wf << U.f1[idx][0] << " " << U.f2[idx][0] << " " << U.f3[idx][0] << "\n";
  }
  
  wf.close();

}

void writeField(std::string filename, Tensor U, int precision){

  // Creat file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file
  for(long idx = 0; idx < N * N * N; idx++){
    wf << U.f11[idx][0] << " " << U.f12[idx][0] << " " << U.f13[idx][0] << " " 
       << U.f22[idx][0] << " " << U.f23[idx][0] << " " << U.f33[idx][0] << "\n";
  }

  wf.close();

}


#endif
