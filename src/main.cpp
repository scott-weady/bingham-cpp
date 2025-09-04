/******************************************************************************
*  Apolar active nematic model in 3D
******************************************************************************/

#include <filesystem>
#include <iostream>
#include <omp.h>
#include <string> 

#include <config.hpp>

// Dimensionless variables (global scope)
double L, sigma_a, sigma_b, zeta, dT, dR;

// Grid resolution (global scope)
int N;

#include <bingham.hpp>
#include <io.hpp>
#include <nematic.hpp>
#include <spectral.hpp>
#include <tensor.hpp>
#include <timesteppers.hpp>
#include <utils.hpp>

int main(int argc, char** argv) {

  auto nthreads = omp_get_max_threads();

  // Check command line arguments
  if(argc < 2) std::cout << "Defaulting to max threads (" << nthreads << ")" << '\n';
  else nthreads = std::min(atoi(argv[1]), nthreads);

  omp_set_num_threads(nthreads);

  // Check if resume, otherwise start from new initial condition
  auto resume = false;
  if(argc == 3) resume = (bool) atoi(argv[2]);

  //************************************************************
  //  Set up io
  //************************************************************
  
  // Output precision
  const int plotPrecision = 4;
  const int savePrecision = 16;

  // Create save directory
  auto results = (std::string) "results";

  // Remove existing directory if it exists
  namespace fs = std::filesystem;
  fs::path results_path = results;

  try {
      if (fs::exists(results_path)) fs::remove_all(results_path);
      fs::create_directory(results_path);
  } catch (const fs::filesystem_error& e) {
      std::cerr << "Filesystem error: " << e.what() << "\n";
  }

  // Create output folders
  createFolder(results + "/Q");
  createFolder(results + "/u");
  createFolder(results + "/Qmag");
  createFolder(results + "/umag");

  // Output file
  auto timeStepLog = results + "/timeStepInfo.txt";

  //************************************************************ 
  //  Load parameters
  //************************************************************ 

  auto p = loadParameters("params.toml");

  // Interpolant resolution
  auto Ncheb = p.res.Ncheb;

  // Assign time stepping parameters
  auto t0 = p.time.t0;
  auto tf = p.time.tf;
  auto tplt = p.time.tplt;
  auto tsave = p.time.tsave;

  auto dt_max = p.time.dt_max;
  auto dt = dt_max, dtm1 = dt_max;
    
  // Grid resolution
  N = p.res.N;

  // Box size
  L = p.dim.L;

  // Stress and alignment coefficients
  sigma_a = p.dim.sigma_a;
  sigma_b = p.dim.sigma_b;
  zeta = p.dim.zeta;
  
  // Diffusion coefficients
  dT = p.dim.dT;
  dR = p.dim.dR;

  // Volume differential
  auto dV = (L * L * L) / (N * N * N);

  //************************************************************ 
  //  Set up FFT
  //************************************************************ 

  std::cout << "Preparing FFT..." << '\n';
  FastFourierTransform fft(N, L, nthreads);

  //************************************************************ 
  //  Set up temporal integrator
  //************************************************************ 

  // Time stepping counters
  auto Nt = (int) std::floor((tf - t0) / dt + 0.1); //number of time steps
  auto Nplt = (int) std::floor(tplt / dt + 0.1); //plot number
  auto Nsave = (int) std::floor(tsave / dt + 0.1); //high resolution save

  auto nplt = 0; //plot counter
  auto nsave = 0; //save counter

  auto t = t0; //initial time
  auto lastTimePlotted = t0; //counter for plotting interval
  auto lastTimeSaved = t0; //counter for saving interval

  // Initialize time stepping operator
  auto timeSteppingOperator = new double[N * N * N];
  helmholtzOperator(timeSteppingOperator, dt * dT, fft); 

  // Clock for timing
  double loopTimer;

  //************************************************************  
  //  Initialize variables 
  //************************************************************ 

  // Preallocate arrays
  auto Qmag = new double[N * N * N];
  auto umag = new double[N * N * N];

  auto u = tensor::zeros1(N);
  auto up1 = tensor::zeros1(N);
  auto Du = tensor::zeros2(N);

  auto Q = tensor::zeros2(N);
  auto Qm1 = tensor::zeros2(N);
  auto DQ = tensor::zeros3(N);

  auto F = tensor::zeros2(N);
  auto Fm1 = tensor::zeros2(N);

  auto Sigma = tensor::zeros2(N);

  auto ST = BinghamClosure(Ncheb, N, zeta);

  //************************************************************ 
  //  Function captures (note these overwrite variables in place)
  //************************************************************ 

  // Closure for computing the fluid velocity
  auto updateVelocity = [&]() {
    fluidSolver(u, Du, Q, Sigma, ST, up1, fft);
    return u;
  };

  // Function for evaluating all nonlinear terms
  auto updateNonlinearity = [&]() {
    updateVelocity(); //velocity
    evaluateNonlinearity(F, u, Du, Q, DQ, ST, fft); //nonlinear terms
    return F;
  };

  auto updateQ = [&](double dt, double dtm1) {
    updateNonlinearity(); //nonlinear terms
    sbdf2(Q, Qm1, F, Fm1, timeSteppingOperator, dt, dtm1, fft, true); //take a time step
    symmetrize(Q); //enforce trace condition and symmetry
  };

  //************************************************************ 
  //  Construct initial data and write
  //************************************************************ 

  initialCondition(Q, resume, fft);

  // Compute quantities for plotting and saving
  updateVelocity();
  
  saveField(results, "umag", nplt, umag, plotPrecision);
  saveField(results, "Qmag", nplt, Qmag, plotPrecision);

  saveField(results, "u", nsave, u, savePrecision);
  saveField(results, "Q", nsave, Q, savePrecision);

  // Update file counter
  nplt++;
  nsave++;

  std::cout << "Initialization complete, starting main loop..." << '\n';

  //************************************************************ 
  //  Euler step to initialize multistep method
  //************************************************************

  // Start loop timer
  loopTimer = omp_get_wtime();
  
  // Evaluate nonlinear terms 
  updateNonlinearity();

  // Store previous step for multistep method
  copy(Qm1, Q); copy(Fm1, F);
  
  // Take first time step with Euler method
  euler(Q, F, timeSteppingOperator, dt, fft, true);
  symmetrize(Q); //enforce trace condition and symmetry

  // Time loop
  loopTimer -= omp_get_wtime();

  // Write time step information to file
  printTimestepInfo(timeStepLog, t, dt, u, dV, -loopTimer);

  // Update time
  t += dt;

  // Update time stepping operator
  helmholtzOperator(timeSteppingOperator, (2.0 / 3.0) * dt * dT, fft); 

  //************************************************************ 
  //  Main loop 
  //************************************************************

  while(t < tf){ 

    // Start loop timer
    loopTimer = omp_get_wtime(); 

    // Take a time step
    updateQ(dt, dtm1);

    // Update time
    t += dt;

    // Stop loop timer
    loopTimer -= omp_get_wtime();

    // End simulation if unstable
    if(std::isnan(Q[0][0][0][0])){
      std::cerr << "Unstable, stopping simulation..." << '\n';
      return 0;  
    }

    // Write time step information to file
    printTimestepInfo(timeStepLog, t, dt, u, dV, -loopTimer);

    // Low resolution file save
    if(t - lastTimePlotted >= tplt){

      // Compute quantities for plotting
      updateVelocity();
      magnitude(u, umag);
      nematicOrderParameter(Q, Qmag);

      saveField(results, "umag", nplt, umag, plotPrecision);
      saveField(results, "Qmag", nplt, Qmag, plotPrecision);

      // Update plotting time and counter
      lastTimePlotted = t;
      nplt++;

    }

    // High resolution file save
    if(t - lastTimeSaved >= tsave){

      updateVelocity();

      saveField(results, "u", nsave, u, savePrecision);
      saveField(results, "Q", nsave, Q, savePrecision);

      // Update saving time and counter
      lastTimeSaved = t;
      nsave++;

    }

    // Update time step through CFL condition
    auto dtp1 = std::min(0.375 * L / (Linf(u) * N), dt_max);

    if(t + dtp1 > lastTimePlotted + tplt) dtp1 = lastTimePlotted + tplt - t; // Ensure we plot on time
    if(t + dtp1 > lastTimePlotted + tf) dtp1 = lastTimePlotted + tf - t; // Ensure we plot at final time
  
    // Time step ratio
    auto r = dtp1 / dt;

    // Update time stepping operator
    helmholtzOperator(timeSteppingOperator, (1 + r) / (1 + 2 * r) * dtp1 * dT, fft); 

    // Prepare for next time step
    dt = dtp1;
    dtm1 = dt;

  }

  // Clean up
  return 0;

}
