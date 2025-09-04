/******************************************************************************

Three-dimensional active nematic fluid, solved with the Bingham closure 

The discretization is pseudo-spectral, and time stepping is done with a
second order imex SBDF2 scheme. To run the code, call

  ./main [nthreads] [resume: 0 or 1 (optional, default 0)]

Dimensionless parameters and resolution should be specified in the params.h file.

******************************************************************************/

#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <cstring> 

const double pi = 3.141592653589793; //pi

const int savePrecision = 16; //resolution for save all
const int plotPrecision = 4; //resolution for visualization

// Global parameters
int N, Ncheb;
double Z, inv_Z;
double L, sigma_a, sigma_b, dT, dR, zeta;

#include "toml.hpp"
#include "cheb_coeffs.h"
#include "utils.hpp"

//********************************************************************
//  Parameter setup
//********************************************************************

// Dimensionless parameters
struct Dimensionless {
    double L, sigma_a, sigma_b, dT, dR, zeta;
};

// Resolution parameters
struct Resolution {
    int N, Ncheb;
};

// Temporal parameters
struct Time {
    double t0, tf, dt_max, tplt, tsave;
};

struct Params {
    Dimensionless dim;
    Resolution res;
    Time time;
};

// Function to load params.toml
Params load_params(const std::string& filename) {

    auto config = toml::parse_file(filename);

    Params p;
    // Dimensionless
    p.dim.L       = config["dimensionless"]["L"].value_or(1.0);
    p.dim.sigma_a = config["dimensionless"]["sigma_a"].value_or(-1.0);
    p.dim.sigma_b = config["dimensionless"]["sigma_b"].value_or(0.0);
    p.dim.dT      = config["dimensionless"]["dT"].value_or(1e-4);
    p.dim.dR      = config["dimensionless"]["dR"].value_or(0.0);
    p.dim.zeta    = config["dimensionless"]["zeta"].value_or(0.0);

    // Resolution
    p.res.N       = config["resolution"]["N"].value_or(64);
    p.res.Ncheb   = config["resolution"]["Ncheb"].value_or(maxChebDegree);

    // Time
    p.time.t0     = config["time"]["t0"].value_or(0.0);
    p.time.tf     = config["time"]["tf"].value_or(200.0);
    p.time.dt_max = config["time"]["dt"].value_or(0.1);
    p.time.tplt   = config["time"]["tplt"].value_or(1.0);
    p.time.tsave  = config["time"]["tsave"].value_or(5.0);

    return p;
}

int main(int argc, char** argv) {

  //************************************************************ 
  // Configure
  //************************************************************ 

  // Initialize number of threads variable
  int nthreads;

  // Indicator for resume
  int resume;

  // Get number of threads
  if(argc < 2){
    std::cerr << " Error: To run code, call\n\n    ./main (nthreads) (resume: optional)\n" << std::endl;
    return 0;
  }
  else{
    nthreads = std::min(atoi(argv[1]), omp_get_max_threads());
  }

  // Check if resume, otherwise start from new initial condition
  if(argc == 3){
    resume = atoi(argv[2]);
  }
  else{
    resume = 0;
  }

  // Set number of threads
  omp_set_num_threads(nthreads);
  fftw_plan_with_nthreads(nthreads);

  //************************************************************
  //  Set up io
  //************************************************************

  std::string message, filename;
  int check;

  // Create save directory
  const std::string outputDirectory = "result";

  // Remove output directory if it exists and create a new one
  message = "rm -rf " + outputDirectory;
  check = system(message.c_str());
  message = "mkdir " + outputDirectory;
  check = system(message.c_str());

  // Create output folders
  createFolder(outputDirectory + "/Q");
  createFolder(outputDirectory + "/s");
  createFolder(outputDirectory + "/Umag");
  createFolder(outputDirectory + "/U");

  // Output file
  std::string output_str = outputDirectory + "/timeStepInfo.txt";
  FILE* output_ptr = fopen(output_str.c_str(),"w");

  //************************************************************ 
  //  Load parameters
  //************************************************************ 

  Params p = load_params("params.toml");
  
  // Grid resolution
  N = p.res.N;

  // Interpolant resolution
  Ncheb = p.res.Ncheb;

  // Assign temporal parameters
  double t0 = p.time.t0;
  double tf = p.time.tf;
  double dt_max = p.time.dt_max;
  double tplt = p.time.tplt;
  double tsave = p.time.tsave;
  double dt = dt_max;

  // Diffusivities
  dT = p.dim.dT;
  dR = p.dim.dR;

  // Other dimensionless parameters
  L = p.dim.L;
  sigma_a = p.dim.sigma_a;
  sigma_b = p.dim.sigma_b;
  zeta = p.dim.zeta;

  // Volume differential
  double dV = (L * L * L) / (N * N * N);

  Z = N * N * N;
  inv_Z = 1.0 / Z;

  if(Ncheb > maxChebDegree){
    std::cerr << "Error: Ncheb exceeds maxChebDegree = " << maxChebDegree << std::endl;
    return 0;
  }

  //************************************************************ 
  //  Set up temporal integrator
  //************************************************************ 

  // Time stepping counters
  int Nt = std::floor((tf - t0) / dt + 0.1); //number of time steps
  int Nplt = (int) std::floor(tplt / dt + 0.1); //plot number
  int Nsave = (int) std::floor(tsave / dt + 0.1); //high resolution save

  int nt = 0; //time step counter
  int nplt = 0; //frame counter
  int nsave = 0; //save counter

  double t = t0; //initial time
  double lastTimePlotted = t0; //counter for plotting interval
  double lastTimeSaved = t0; //counter for saving interval

  // Fluid solve parameters for fixed point iteration
  double fluidSolverTolerance = 1e-8;
  int maxFluidSolverIterations = 10;
  
  // Preallocate time-stepping arrays
  double* EulerOperator = new double[N * N * N];
  double* SBDF2Operator = new double[N * N * N];

  // Preallocate wavenumber array
  double* wavenumber = new double[N];

  // Initialize array of Fourier modes
  for(int n = 0; n < N; n++){
    if(n <= N / 2) wavenumber[n] = 2.0 * pi * n / L;
    else           wavenumber[n] = 2.0 * pi * (n - N) / L;
  }

  // Maximum wave number
  const double kmax = 2.0 * pi * (N / 2) / L;

  // Compute initial time stepping arrays
  helmholtzOperator(EulerOperator, wavenumber, dt * dT); 
  helmholtzOperator(SBDF2Operator, wavenumber, (2.0 / 3.0) * dt * dT); 

  // Clock for timing
  double loopTimer;

  //************************************************************ 
  //  Setup FFT
  //************************************************************ 

  std::cout << "Preparing FFT..." << std::endl;
  FastFourierTransform fft(N);

  //************************************************************  
  //  Initialize variables 
  //************************************************************ 

  double* s = new double[N * N * N];
  double* Umag = new double[N * N * N];
  
  // Boolean indicator for symmetric tensor
  bool symmetric = true;

  // Initialize velocity
  fftw_complex *u[3], *u_h[3], *up1[3], *up1_h[3], *buffer[3];
  for(auto i = 0; i < 3; i++){
    u[i]      = zeros(Z);
    u_h[i]    = zeros(Z);
    up1[i]    = zeros(Z);
    up1_h[i]  = zeros(Z);
    buffer[i] = zeros(Z);
  }

  // Initialize nematic tensor
  fftw_complex *Q[3][3], *Q_h[3][3], *Qm1_h[3][3];
  fftw_complex *F[3][3], *F_h[3][3], *Fm1_h[3][3];
  fftw_complex *Sigma[3][3], *Sigma_h[3][3], *ST[3][3];
  
  for(auto i = 0; i < 3; i++){
    for(auto j = i; j < 3; j++){

      Q[i][j]     = zeros(Z);
      Q_h[i][j]   = zeros(Z);
      Qm1_h[i][j] = zeros(Z);

      F[i][j]     = zeros(Z);
      F_h[i][j]   = zeros(Z);
      Fm1_h[i][j] = zeros(Z);

      Sigma[i][j]   = zeros(Z);
      Sigma_h[i][j] = zeros(Z);
      ST[i][j]      = zeros(Z);
      
      if(i != j){
        Q[j][i]     = Q[i][j];
        Q_h[j][i]   = Q_h[i][j];
        Qm1_h[j][i] = Qm1_h[i][j];
        F[j][i]     = F[i][j];
        F_h[j][i]   = F_h[i][j];
        Fm1_h[j][i] = Fm1_h[i][j];
        Sigma[j][i]   = Sigma[i][j];
        Sigma_h[j][i] = Sigma_h[i][j];
        ST[j][i]      = ST[i][j];
      }

    }
  }

  fftw_complex* gradU[3][3];
  for(auto i = 0; i < 3; i++){
    for(auto j = 0; j < 3; j++){
      gradU[i][j] = zeros(Z);
    }
  }

  fftw_complex* gradQ[3][3][3];
  for(auto i = 0; i < 3; i++){
    for(auto j = i; j < 3; j++){
      for(auto k = 0; k < 3; k++){
        gradQ[i][j][k] = zeros(Z);
        if(i != j){
          gradQ[j][i][k] = gradQ[i][j][k];
        }
      }
    }
  }

  //************************************************************ 
  //  Function captures for velocity and nonlinearity
  //************************************************************ 

  // Closure for computing the fluid velocity
  auto evaluateVelocity = [&](fftw_complex* (&Q)[3][3], fftw_complex* (&u)[3]) {
    fluidSolver(u, u_h, up1, up1_h, Q, Q_h, Sigma, Sigma_h, ST, gradU, buffer, wavenumber, fluidSolverTolerance, maxFluidSolverIterations, fft);
  };

  // Closure for evaluating nonlinear terms
  auto evaluateNonlinearity = [&](fftw_complex* (&Q)[3][3], fftw_complex* (&F)[3][3]) {

    // Solve for fluid velocity
    evaluateVelocity(Q, u);

    // Compute velocity gradient
    grad(u, u_h, gradU, buffer, wavenumber, fft);

    // Evaluate Bingham closure
    binghamClosure(ST, Q, gradU);

    // Compute nematic tensor gradient
    grad(Q, Q_h, gradQ, buffer, wavenumber, fft);

    // Evaluate nonlinear terms
    nonlinear(F, u, gradU, Q, gradQ, ST);

  };

  //************************************************************ 
  //  Construct initial data
  //************************************************************ 

  std::cout << "Initializing Q tensor..." << std::endl;

  // Locate initial data
  if(resume == 1){
    
    std::ifstream Q_init("initial_data/Q.dat");

    if(!Q_init){
      std::cerr << "Error: No initial data found\n" << std::endl;
      return 0;
    }

    double q11, q12, q13;
    double      q22, q23;
    double           q33;

    std::cout << "Initial data found, loading Q..." << std::endl;

    // Line counter
    auto idx = 0;

    // Load
    while (Q_init >> q11 >> q12 >> q13 >> q22 >> q23 >> q33) {

      // Assign values
      if(idx < N * N * N){
        Q[0][0][idx][0] = q11, Q[0][1][idx][0] = q12, Q[0][2][idx][0] = q13;
        Q[1][1][idx][0] = q22, Q[1][2][idx][0] = q23;
        Q[2][2][idx][0] = q33;
      }

      idx++;

    }

    // Check if correct dimensions
    if(idx != N * N * N){
      std::cerr << "Error: Incompatible resolution, N = " << cbrt(idx) << std::endl;
      Q_init.close();
      return 0;
    }

    Q_init.close();

  }
  // If no initial data, apply plane-wave perturbation
  else {

    // Start with isotropic base state
    #pragma omp parallel for
    for(auto idx = 0; idx < N * N * N; idx++){
      Q[0][0][idx][0] = 1.0 / 3.0;
      Q[1][1][idx][0] = 1.0 / 3.0;
      Q[2][2][idx][0] = 1.0 / 3.0;
    }

    // Apply plane wave perturbation
    for(auto i = 0; i < 3; i++){
      for(auto j = i; j < 3; j++){
        perturbation(Q[i][j], wavenumber);

        if(i != j){
          Q[j][i] = Q[i][j];
        }

      }
    }
    
  }

  // Enfore trace condition on Q
  retrace(Q);

  // Store initial data at previous time step for multistep method
  fft.fft(Q, Qm1_h);

  //************************************************************ 
  //  Write initial data
  //************************************************************ 

  // Low resolution scalar nematic order parameter
  nematicOrderParameter(Q, s); 
  filename = outputDirectory + "/s/s-" + std::to_string(nplt) + ".dat"; 
  writeField(filename, s, plotPrecision);      

  // Velocity field
  evaluateVelocity(Q, u);
      
  // Low resolution velocity magnitude
  filename = outputDirectory + "/Umag/Umag-" + std::to_string(nplt) + ".dat";
  magnitude(u, Umag);
  writeField(filename, Umag, plotPrecision);

  // High resolution velocity
  filename = outputDirectory + "/U/U-" + std::to_string(nsave) + ".dat";
  writeField(filename, u, savePrecision);

  // High resolution nematic tensor
  filename = outputDirectory + "/Q/Q-" + std::to_string(nsave) + ".dat"; 
  writeField(filename, Q, savePrecision);

  // Update file counter
  nplt++;
  nsave++;

  std::cout << "Initialization complete, starting main loop..." << std::endl;

  //************************************************************ 
  //  Euler step to initialize multistep method
  //************************************************************

  // Start loop timer
  loopTimer = omp_get_wtime();
  
  // Evaluate nonlinear terms 
  evaluateNonlinearity(Q, F);
  fft.fft(F, Fm1_h); 
  fft.fft(F, F_h); 

  // Antialias (Fourier space only)
  antialias(Fm1_h, wavenumber, kmax);
  antialias(F_h, wavenumber, kmax);

  // Take an Euler step
  fft.fft(Q, Q_h); 
  euler(Q_h, F_h, EulerOperator, dt);
  fft.ifft(Q_h, Q);

  // Enforce trace condition
  retrace(Q);

  // Time loop
  loopTimer = omp_get_wtime() - loopTimer;
  
  // Print out time step information 
  printf("           t = %3.3f \n"
         "          dt = %3.3f \n"
         "     ||U||_2 = %1.14e \n"
         "   ||U||_inf = %1.14e \n"
         "        loop = %1.4es\n"
         "---------------------------\n", t, dt, norm_2(u, dV), norm_inf(u), loopTimer);

  // Write time step information to file
  output_ptr = fopen(output_str.c_str(), "a");    
  fprintf(output_ptr,
         "           t = %3.3f \n"
         "          dt = %3.3f \n"
         "     ||U||_2 = %1.4e \n"
         "   ||U||_inf = %1.4e \n"
         "        loop = %1.4es\n"
         "---------------------------\n", t, dt, norm_2(u, dV), norm_inf(u), loopTimer);

  // Update time
  t += dt;

  fclose(output_ptr);

  //************************************************************ 
  //  Main loop 
  //************************************************************

  double dtm1 = dt;
  double dtp1;

  while(t < tf){ 

    // Start loop timer
    loopTimer = omp_get_wtime();

    // Evaluate nonlinear terms 
    evaluateNonlinearity(Q, F);

    // Antialias (nonlinearity only)
    fft.fft(F, F_h);
    antialias(F_h, wavenumber, kmax);

    // Update nematic tensor
    fft.fft(Q, Q_h);
    sbdf2(Q_h, Qm1_h, F_h, Fm1_h, SBDF2Operator, dt, dtm1);
    fft.ifft(Q_h, Q);

    // Enforce trace condition
    retrace(Q);

    // Time loop
    loopTimer = omp_get_wtime() - loopTimer;

    // End simulation if unstable
    if(std::isnan(Q[0][0][0][0])){
      std::cerr << "Unstable, stopping simulation..." << std::endl;
      return 0;  
    }

    // Update time
    t += dt;

    // Print out time step information 
    printf("           t = %3.3f \n"
           "          dt = %3.3f \n"
           "     ||U||_2 = %1.14e \n"
           "   ||U||_inf = %1.14e \n"
           "        loop = %1.4es\n"
           "---------------------------\n", t, dt, norm_2(u, dV), norm_inf(u), loopTimer);

    // Write time step information to file
    output_ptr = fopen(output_str.c_str(), "a");    
    fprintf(output_ptr,
           "           t = %3.3f \n"
           "          dt = %3.3f \n"
           "     ||U||_2 = %1.4e \n"
           "   ||U||_inf = %1.4e \n"
           "        loop = %1.4es\n"
           "---------------------------\n", t, dt, norm_2(u, dV), norm_inf(u), loopTimer);
    fclose(output_ptr);

  
    // Low resolution file save
    if(t - lastTimePlotted >= tplt){

      // Compute scalar nematic order parameter
      nematicOrderParameter(Q, s);

      // Write nematic order parameter (low resolution)
      filename = outputDirectory + "/s/s-" + std::to_string(nplt) + ".dat"; 
      writeField(filename, s, plotPrecision);      

      // Compute velocity
      evaluateVelocity(Q, u);

      // Computed velocity magnitude
      magnitude(u, Umag);

      // Write velocity magnitude (low resolution)
      filename = outputDirectory + "/Umag/Umag-" + std::to_string(nplt) + ".dat";
      writeField(filename, Umag, plotPrecision);

      // Update plotting time
      lastTimePlotted = t;

      // Update plot counter
      nplt++;

    }

    // High resolution file save
    if(t - lastTimeSaved >= tsave){

      // Compute velocity
      evaluateVelocity(Q, u);

      // Write velocity (high resolution)
      filename = outputDirectory + "/U/U-" + std::to_string(nsave) + ".dat";
      writeField(filename, u, savePrecision);

      // Write nematic tensor (high resolution)
      filename = outputDirectory + "/Q/Q-" + std::to_string(nsave) + ".dat"; 
      writeField(filename, Q, savePrecision);

      lastTimeSaved = t;

      // Update save counter
      nsave++;

    }

    // Compute maximal Courant number
    double CFL = L / (norm_inf(u) * N);

    // Update time step
    dtp1 = tplt / std::round(tplt / std::min(tplt, (3.0 / 8.0) * CFL));
    dtp1 = std::min(dtp1, dt_max);

    // Ensure plotting 
    if(t + dtp1 > lastTimePlotted + tplt){
      dtp1 = lastTimePlotted + tplt - t;
    }

    // If final time, land exactly
    if(t + dtp1 > lastTimePlotted + tf){
      dtp1 = lastTimePlotted + tf - t;
    }

    dt = dtp1;
    dtm1 = dt;

    // Time step ratio
    double r = dt / dtm1;

    // Update time stepping operator
    helmholtzOperator(SBDF2Operator, wavenumber, (1 + r) / (1 + 2 * r) * dt * dT); 

  }

  //************************************************************
  //  Clean up
  //************************************************************

  delete[] wavenumber; 
  delete[] EulerOperator; 
  delete[] SBDF2Operator; 
  delete[] s;
  delete[] Umag;
  
  return 0;

}
