
#pragma once

#include <string>
#include <toml.hpp>

// Dimensionless parameters
struct Dimensionless {
  double L, sigma_a, sigma_b, zeta, dT, dR;
};

// Resolution parameters
struct Resolution {
  int N, Ncheb;
};

// Temporal parameters
struct Time {
  double t0, tf, dt_max, tplt, tsave;
};

// Aggregate parameters
struct Params {
  Dimensionless dim;
  Resolution res;
  Time time;
};

// Function to load parameters from a TOML file
Params loadParameters(const std::string& filename);
