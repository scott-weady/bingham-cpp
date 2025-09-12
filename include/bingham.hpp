
#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <config.hpp>
#include <tensor.hpp>

class BinghamClosure{

public:

    int N; //grid size
    tensor::Tensor2 ST; //contraction of 4th moment with rotation tensor
    Params p; //parameters

    // Constructor
    BinghamClosure(int N, Params p) : N(N), p(p){

        Ncheb = p.res.Ncheb;

        if(Ncheb > maxChebDegree){
            std::cout << "Number of requested modes exceeds maximum degree (" << maxChebDegree << "), " << "defaulting to max." << '\n';
            Ncheb = maxChebDegree;
        }

        C11 = load("include/bingham_coeffs/C11.dat");
        C12 = load("include/bingham_coeffs/C12.dat");
        C22 = load("include/bingham_coeffs/C22.dat");
        ST = tensor::zeros2(N, true); //symmetric

    }

    // Destructor
    ~BinghamClosure(){}

    // Map from Q, Du to S:T
    tensor::Tensor2 compute(tensor::Tensor2 Q, tensor::Tensor2 Du);
    
private:

    // Coefficients for Chebyshev expansion
    std::vector<double> C11;
    std::vector<double> C12;
    std::vector<double> C22;

    // Solver parameters
    double tolerance = 1e-14;
    int maxIterations = 50;

    // Number of Chebyshev modes
    int Ncheb;

    // Maximum degree of Chebyshev expansion
    const int maxChebDegree = 101;

    // Function to load coefficients from .dat file
    std::vector<double> load(std::string filename){

        std::vector<double> Cij;
        std::ifstream file(filename);

        if(!file){
            std::cerr << "Error: Could not open file " << filename << '\n';
            return Cij;
        }

        double value;

        while(file >> value) Cij.push_back(value);

        return Cij;

    }

};
