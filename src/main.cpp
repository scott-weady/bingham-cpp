/******************************************************************************
*  Apolar active nematic model in 3D
******************************************************************************/


// Dimensionless variables (global scope)
double L, sigma_a, sigma_b, zeta, dT, dR;

// Grid resolution (global scope)
int N;

#include <filesystem>
#include <iostream>
#include <omp.h>
#include <string> 

#include <config.hpp>

#include <io.hpp>

#include <nematic.hpp>
#include <integrators.hpp>
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
  SpectralSolver solver(N, L, p, nthreads);

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

  // Clock for timing
  double loopTimer;

  //************************************************************  
  //  Initialize variables 
  //************************************************************ 

  // Preallocate arrays
  auto Qmag = tensor::zeros(N);
  auto umag = tensor::zeros(N);

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
    fluidSolver(u, Du, Q, Sigma, ST, up1, solver);
    return u;
  };

  // Function for evaluating all nonlinear terms
  auto updateNonlinearity = [&]() {
    updateVelocity(); //velocity
    evaluateNonlinearity(F, u, Du, Q, DQ, ST, solver); //nonlinear terms
    return F;
  };

  auto updateQ = [&](double dt, double dtm1) {
    updateNonlinearity(); //nonlinear terms
    sbdf2(Q, Qm1, F, Fm1, dt, dtm1, solver, true); //take a time step
    symmetrize(Q); //enforce trace condition and symmetry
  };

  auto plot = [&]() {
    updateVelocity();
    magnitude(u, umag);
    nematicOrderParameter(Q, Qmag);
    saveField(results, "umag", nplt, umag, plotPrecision);
    saveField(results, "Qmag", nplt, Qmag, plotPrecision);
  };

  auto save = [&]() {
    updateVelocity();
    saveField(results, "u", nsave, u, savePrecision);
    saveField(results, "Q", nsave, Q, savePrecision);
  };

  //************************************************************ 
  //  Construct initial data and write
  //************************************************************ 

  initialCondition(Q, resume, solver);

  plot();
  save();

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
  euler(Q, F, dt, solver, true);
  symmetrize(Q); //enforce trace condition and symmetry

  // Time loop
  loopTimer -= omp_get_wtime();

  // Write time step information to file
  printTimestepInfo(timeStepLog, t, dt, u, dV, -loopTimer);

  // Update time
  t += dt;

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
    if(t - lastTimePlotted == tplt){
      plot();
      lastTimePlotted = t;
      nplt++;
    }

    // High resolution file save
    if(t - lastTimeSaved == tsave){
      save();
      lastTimeSaved = t;
      nsave++;
    }

    // Update time step through CFL condition
    dtm1 = dt;
    dt = std::min(0.375 * L / (Linf(u) * N), dt_max);

    if(t + dt > lastTimePlotted + tplt) dt = lastTimePlotted + tplt - t; // Ensure we plot on time
    if(t + dt > lastTimeSaved + tsave) dt = lastTimeSaved + tsave - t; // Ensure we save on time
    if(t + dt > lastTimePlotted + tf) dt = lastTimePlotted + tf - t; // Ensure we plot at final time
  
  }

  // Clean up
  return 0;

}
