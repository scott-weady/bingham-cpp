
#pragma once

#include <iostream>
#include <string>
#include <type_traits>

#include <tensor.hpp>
#include <utils.hpp>

/** Create directory
 * @param foldername Name of directory to create
 */
auto createFolder(std::string foldername){
  auto message = "mkdir " + foldername;
  return system(message.c_str());
}

/** Save field to output directory
 * @param outputDir Output directory
 * @param name Name of the field
 * @param nsave Save index
 * @param field Field data
 * @param precision Floating point precision
 */
template <typename T>
auto saveField(const std::string& outputDir, const std::string& name, int nsave, T& field,int precision){

  auto filename = outputDir + "/" + name + "/" + name + "-" + std::to_string(nsave) + ".dat";

  // Create file
  std::ofstream wf(filename, std::ios::out | std::ios::binary);

  // Set precision
  wf.precision(precision); 

  // Write to file based on type
  for(auto idx = 0; idx < N * N * N; idx++){

    if constexpr(std::is_same_v<T, fftw_complex*>) {
      wf << field[idx][0] << "\n";
    } 
    else if constexpr (std::is_same_v<T, tensor::Tensor1>) {
      wf << field[0][idx][0] << " " << field[1][idx][0] << " " << field[2][idx][0] << "\n";
    } 
    else if constexpr (std::is_same_v<T, tensor::Tensor2>) {
      wf << field[0][0][idx][0] << " " << field[0][1][idx][0] << " " << field[0][2][idx][0] << " "
         << field[1][1][idx][0] << " " << field[1][2][idx][0] << " " << field[2][2][idx][0] << "\n";
    }
    
  }

}

/** Print timestep information to console and file
 * @param timeStepLog Log file path
 * @param t Current time
 * @param dt Time step
 * @param u Velocity field
 * @param dV Volume element
 * @param loopTimer Loop time
 */
auto printTimestepInfo(std::string timeStepLog, double t, double dt, tensor::Tensor1 u, double dV, double loopTimer) {

  std::ofstream wf(timeStepLog, std::ios::out | std::ios::app);
  std::ostringstream msg;

  msg << "           t = " << t << '\n'
      << "          t = " << dt << '\n'
      << "     ||U||₂ = " << L2(u, dV) << '\n'
      << "   ||U||∞ = " << Linf(u) << '\n'
      << "        loop = " << loopTimer << "s\n"
      << "---------------------------\n";

  // Send to console and file
  std::cout << msg.str();
  wf << msg.str();

}