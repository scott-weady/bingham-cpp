
#include <config.hpp>
#include <toml.hpp>

Params loadParameters(const std::string& filename) {

  auto config = toml::parse_file(filename);

  Params p;
  
  // Dimensionless
  p.dim.L       = config["dimensionless"]["L"].value_or(1.0);
  p.dim.sigma_a = config["dimensionless"]["sigma_a"].value_or(-1.0);
  p.dim.sigma_b = config["dimensionless"]["sigma_b"].value_or(0.0);
  p.dim.zeta    = config["dimensionless"]["zeta"].value_or(0.0);
  p.dim.dT      = config["dimensionless"]["dT"].value_or(1e-3);
  p.dim.dR      = config["dimensionless"]["dR"].value_or(0.0);

  // Resolution
  p.res.N       = config["resolution"]["N"].value_or(64);
  p.res.Ncheb   = config["resolution"]["Ncheb"].value_or(101);

  // Time
  p.time.t0     = config["time"]["t0"].value_or(0.0);
  p.time.tf     = config["time"]["tf"].value_or(200.0);
  p.time.dt_max = config["time"]["dt"].value_or(0.1);
  p.time.tplt   = config["time"]["tplt"].value_or(1.0);
  p.time.tsave  = config["time"]["tsave"].value_or(5.0);

  return p;

}