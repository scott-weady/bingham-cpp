
#pragma once

#include <iostream>
#include <string>
#include <tensor.hpp>
#include <utils.hpp>


// Create directory
auto createFolder(std::string foldername){
  auto message = "mkdir " + foldername;
  return system(message.c_str());
}

// Save field to output directory
template <typename T>
auto saveField(const std::string& outputDir, const std::string& name, int nsave, T& field,int precision){

  auto filename = outputDir + "/" + name + "/" + name + "-" + std::to_string(nsave) + ".dat";

  // Create file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file based on type
  for(auto idx = 0; idx < N * N * N; idx++){

    if constexpr (std::is_same<T, double*>::value) {
      wf << field[idx] << "\n";
    } 
    else if constexpr(std::is_same<T, fftw_complex*>::value) {
      wf << field[idx][0] << "\n";
    } 
    else if constexpr (std::is_same<T, tensor::Tensor1>::value) {
      wf << field[0][idx][0] << " " << field[1][idx][0] << " " << field[2][idx][0] << "\n";
    } 
    else if constexpr (std::is_same<T, tensor::Tensor2>::value) {
      wf << field[0][0][idx][0] << " " << field[0][1][idx][0] << " " << field[0][2][idx][0] << " "
         << field[1][1][idx][0] << " " << field[1][2][idx][0] << " " << field[2][2][idx][0] << "\n";
    }
    
  }

}

// Print timestep information to console and file
auto printTimestepInfo(std::string timeStepLog, double t, double dt, tensor::Tensor1 u, double dV, double loopTimer) {

  std::ofstream wf(timeStepLog, std::ios::out | std::ios::app);
  std::ostringstream msg;
  
  msg << "           t = " << t << '\n'
      << "          dt = " << dt << '\n'
      << "     ||U||_2 = " << L2(u, dV) << '\n'
      << "   ||U||_inf = " << Linf(u) << '\n'
      << "        loop = " << loopTimer << "s\n"
      << "---------------------------\n";

  // Send to console and file
  std::cout << msg.str();
  wf << msg.str();

}