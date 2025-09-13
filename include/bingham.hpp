
#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include <config.hpp>
#include <tensor.hpp>

class BinghamClosure{

public:

    int N; //grid size
    tensor::Tensor2 ST; //contraction of 4th moment with rotation tensor
    Params p; //parameters
    static constexpr int maxChebDegree = 101; //maximum degree of Chebyshev expansion
    int Ncheb; //degree of Chebyshev expansion

    // Coefficients for Chebyshev expansion
    static std::array<double, maxChebDegree * maxChebDegree> C11;
    static std::array<double, maxChebDegree * maxChebDegree> C12;
    static std::array<double, maxChebDegree * maxChebDegree> C22;

    // Constructor
    BinghamClosure(int N, Params p) : N(N), p(p){

        Ncheb = p.res.Ncheb;

        if(Ncheb > maxChebDegree){
            std::cout << "Number of requested modes exceeds maximum degree (" << maxChebDegree << "), " << "defaulting to max." << '\n';
            Ncheb = maxChebDegree;
        }

        ST = tensor::zeros2(N, true); //symmetric

    }

    // Destructor
    ~BinghamClosure(){}

    // Map from Q, Du to S:T
    tensor::Tensor2 compute(tensor::Tensor2 Q, tensor::Tensor2 Du);
    
private:

    // Solver parameters
    double tolerance = 1e-14;
    int maxIterations = 50;

};
