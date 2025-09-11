
#include <bingham.hpp>

/** Compute Bingham closure
 * @param Q Nematic tensor
 * @param Du Velocity gradient
 * @return ST Extra stress tensor
 */
tensor::Tensor2 BinghamClosure::compute(tensor::Tensor2 Q, tensor::Tensor2 Du){
        
    auto zeta = p.dim.zeta; //alignment parameter

    auto converged = 1; //flag to check convergence

    // Loop over all grid points
    #pragma omp parallel for reduction (*:converged)
    for(auto idx = 0; idx < N * N * N; idx++){

        // Local copy of the Q Tensor (perturb to avoid degenerate eigenvector)
        auto Q11 = Q[0][0][idx][0];
        auto Q12 = Q[0][1][idx][0] + 1e-15;
        auto Q13 = Q[0][2][idx][0] + 1e-15;
        auto Q22 = Q[1][1][idx][0];
        auto Q23 = Q[1][2][idx][0] + 1e-15;
        auto Q33 = Q[2][2][idx][0];

        // Concentration
        auto c = Q11 + Q22 + Q33; 

        // Normalize
        Q11 /= c;
        Q12 /= c;
        Q13 /= c;
        Q22 /= c;
        Q23 /= c;
        Q33 /= c;

        // Get gradient of velocity
        auto E11 = Du[0][0][idx][0]; 
        auto E12 = 0.5 * (Du[0][1][idx][0] + Du[1][0][idx][0]); 
        auto E13 = 0.5 * (Du[0][2][idx][0] + Du[2][0][idx][0]);
        auto E22 = Du[1][1][idx][0];
        auto E23 = 0.5 * (Du[1][2][idx][0] + Du[2][1][idx][0]);
        auto E33 = Du[2][2][idx][0];

        // Construct and solve characteristic polynomial
        auto mu = (1.0 / 3.0); //initial guess
        auto a0 = Q11 * Q23 * Q23 + Q22 * Q13 * Q13 + Q33 * Q12 * Q12 - Q11 * Q22 * Q33 - 2 * Q13 * Q12 * Q23;
        auto a1 = Q11 * Q22 + Q11 * Q33 + Q22 * Q33 - (Q12 * Q12 + Q13 * Q13 + Q23 * Q23);
    
        // Initial function evaluation
        auto chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;
        
        // Start iteration count
        auto iteration = 0;

        // Solve for one eigenvalue using Newton's method
        while(std::abs(chi_mu) > tolerance && iteration < maxIterations){

            // Evaluate function
            chi_mu = mu * mu * mu - mu * mu + a1 * mu + a0;

            // Compute update
            mu += -chi_mu / (3 * mu * mu - 2 * mu + a1);

            // Update iteration count
            iteration++;

        }

        if (iteration == maxIterations){
            converged *= 0;
        }

        // Get other eigenvalues
        auto nu1 = mu;
        auto nu2 = 0.5 * (-(mu - 1) + std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
        auto nu3 = 0.5 * (-(mu - 1) - std::sqrt(std::abs((mu - 1) * (mu - 1) - 4.0 * (a1 + mu * (mu - 1)))));
        
        // Sort
        auto mu1 = std::max(std::max(nu1, nu2), nu3);    
        auto mu3 = std::min(std::min(nu1, nu2), nu3);    
        auto mu2 = 1.0 - (mu1 + mu3);
        
        // First eigenvector
        auto O11 = Q12 * Q23 - Q13 * (Q22 - mu1);
        auto O21 = Q13 * Q12 - (Q11 - mu1) * Q23;
        auto O31 = (Q11 - mu1) * (Q22 - mu1) - Q12 * Q12;

        // Normalize
        auto m1 = std::sqrt(O11 * O11 + O21 * O21 + O31 * O31);
        O11 /= m1; 
        O21 /= m1; 
        O31 /= m1;

        // Second eigenvector
        auto O12 = Q12 * Q23 - Q13 * (Q22 - mu2);
        auto O22 = Q13 * Q12 - (Q11 - mu2) * Q23;
        auto O32 = (Q11 - mu2) * (Q22 - mu2) - Q12 * Q12;

        // Normalize
        auto m2 = std::sqrt(O12 * O12 + O22 * O22 + O32 * O32);
        O12 /= m2; 
        O22 /= m2; 
        O32 /= m2;

        // Third eigenvector
        auto O13 = O21 * O32 - O31 * O22;
        auto O23 = O31 * O12 - O11 * O32;
        auto O33 = O11 * O22 - O21 * O12;

        // Normalize
        auto m3 = std::sqrt(O13 * O13 + O23 * O23 + O33 * O33);
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

        // Domain transformation
        auto pmu1 = (mu1 - 1.0 / 3.0) - (mu2 - 1.0 / 3.0);
        auto pmu2 = 2 * (mu1 - 1.0 / 3.0) + 4 * (mu2 - 1.0 / 3.0);

        // Shift to [-1, 1]^2
        nu1 = 2 * std::min(std::max((pmu1 + pmu2), 0.0), 1.0) - 1; 
        nu2 = 2 * std::min(std::max(pmu1 / (pmu1 + pmu2), 0.0), 1.0) - 1;
        
        auto Tim2 = 1.0, Tjm2 = 1.0;
        auto Tim1 = nu1, Tjm1 = nu2;
        double Ti, Tj;

        // Evaluate coefficients (0, 0), (0, 1), (1, 0), (1, 1)
        auto tS1111 = Tim2 * (C11[0] * Tjm2 + C11[0 + maxChebDegree] * Tjm1) + Tim1 * (C11[1] * Tjm2 + C11[1 + maxChebDegree] * Tjm1);
        auto tS1122 = Tim2 * (C12[0] * Tjm2 + C12[0 + maxChebDegree] * Tjm1) + Tim1 * (C12[1] * Tjm2 + C12[1 + maxChebDegree] * Tjm1);
        auto tS2222 = Tim2 * (C22[0] * Tjm2 + C22[0 + maxChebDegree] * Tjm1) + Tim1 * (C22[1] * Tjm2 + C22[1 + maxChebDegree] * Tjm1);

        // Evaluate coefficients along (:, 0) and (:, 1)
        for(auto nx = 2; nx < Ncheb; nx++){

            Ti = 2 * nu1 * Tim1 - Tim2;

            tS1111 += Ti * (C11[nx] * Tjm2 + C11[nx + maxChebDegree] * Tjm1);
            tS1122 += Ti * (C12[nx] * Tjm2 + C12[nx + maxChebDegree] * Tjm1);
            tS2222 += Ti * (C22[nx] * Tjm2 + C22[nx + maxChebDegree] * Tjm1);

            Tim2 = Tim1; Tim1 = Ti;

        }      

        // Evaluate via recursion relation
        for(auto ny = 2; ny < Ncheb; ny++){

            Tj = 2 * nu2 * Tjm1 - Tjm2;

            Tim2 = 1.0; Tim1 = nu1;

            tS1111 += Tim2 * C11[0 + maxChebDegree * ny] * Tj + Tim1 * C11[1 + maxChebDegree * ny] * Tj;
            tS1122 += Tim2 * C12[0 + maxChebDegree * ny] * Tj + Tim1 * C12[1 + maxChebDegree * ny] * Tj;
            tS2222 += Tim2 * C22[0 + maxChebDegree * ny] * Tj + Tim1 * C22[1 + maxChebDegree * ny] * Tj;

            for(auto nx = 2; nx < (Ncheb - ny); nx++){

                Ti = 2 * nu1 * Tim1 - Tim2;

                tS1111 += Ti * C11[nx + maxChebDegree * ny] * Tj;
                tS1122 += Ti * C12[nx + maxChebDegree * ny] * Tj;
                tS2222 += Ti * C22[nx + maxChebDegree * ny] * Tj;

                Tim2 = Tim1; Tim1 = Ti;

            }

            Tjm2 = Tjm1; Tjm1 = Tj;

        }

        // Evaluate remaining terms via trace identities
        auto tS1133 = mu1 - tS1111 - tS1122;
        auto tS2233 = mu2 - tS1122 - tS2222;
        auto tS3333 = mu3 - tS1133 - tS2233;

        // Compute rotations
        auto tT11 = O11 * (E11 * O11 + 2 * E12 * O21 + 2 * E13 * O31) + O21 * (E22 * O21 + 2 * E23 * O31) + E33 * O31 * O31 + 2 * zeta * mu1;   
        auto tT12 = E11 * O11 * O12 + E12 * (O11 * O22 + O21 * O12) + E22 * O21 * O22 + E13 * (O11 * O32 + O31 * O12) + E33 * O31 * O32 + E23 * (O21 * O32 + O31 * O22);
        auto tT13 = E11 * O11 * O13 + E12 * (O11 * O23 + O21 * O13) + E13 * (O11 * O33 + O31 * O13) + E22 * O21 * O23 + E23 * (O21 * O33 + O31 * O23) + E33 * O31 * O33;
        auto tT22 = O12 * (E11 * O12 + 2 * E12 * O22 + 2 * E13 * O32) + O22 * (E22 * O22 + 2 * E23 * O32) + E33 * O32 * O32 + 2 * zeta * mu2;  
        auto tT23 = E11 * O12 * O13 + E12 * (O12 * O23 + O22 * O13) + E13 * (O12 * O33 + O32 * O13) + E22 * O22 * O23 + E23 * (O22 * O33 + O32 * O23) + E33 * O32 * O33;
        auto tT33 = O13 * (E11 * O13 + 2 * E12 * O23 + 2 * E13 * O33) + O23 * (E22 * O23 + 2 * E23 * O33) + E33 * O33 * O33 + 2 * zeta * mu3;  
            
        // Compute contraction in rotated frame
        auto tST11 = tS1111 * tT11 + tS1122 * tT22 + tS1133 * tT33;
        auto tST12 = 2 * tS1122 * tT12;
        auto tST13 = 2 * tS1133 * tT13;
        auto tST22 = tS1122 * tT11 + tS2222 * tT22 + tS2233 * tT33;
        auto tST23 = 2 * tS2233 * tT23;
        auto tST33 = tS1133 * tT11 + tS2233 * tT22 + tS3333 * tT33;

        // Rotate to physical frame and store
        ST[0][0][idx][0] = c * (O11 * (O11 * tST11 + O12 * tST12 + O13 * tST13) + O12 * (O11 * tST12 + O12 * tST22 + O13 * tST23) + O13 * (O11 * tST13 + O12 * tST23 + O13 * tST33));
        ST[0][1][idx][0] = c * (O11 * (O21 * tST11 + O22 * tST12 + O23 * tST13) + O12 * (O21 * tST12 + O22 * tST22 + O23 * tST23) + O13 * (O21 * tST13 + O22 * tST23 + O23 * tST33));
        ST[0][2][idx][0] = c * (O11 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + O12 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + O13 * (O31 * tST13 + O32 * tST23 + O33 * tST33));
        ST[1][1][idx][0] = c * (O21 * (O21 * tST11 + O22 * tST12 + O23 * tST13) + O22 * (O21 * tST12 + O22 * tST22 + O23 * tST23) + O23 * (O21 * tST13 + O22 * tST23 + O23 * tST33));
        ST[1][2][idx][0] = c * (O21 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + O22 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + O23 * (O31 * tST13 + O32 * tST23 + O33 * tST33));
        ST[2][2][idx][0] = c * (O31 * (O31 * tST11 + O32 * tST12 + O33 * tST13) + O32 * (O31 * tST12 + O32 * tST22 + O33 * tST23) + O33 * (O31 * tST13 + O32 * tST23 + O33 * tST33));
        ST[1][0][idx][0] = ST[0][1][idx][0];
        ST[2][0][idx][0] = ST[0][2][idx][0];
        ST[2][1][idx][0] = ST[1][2][idx][0];

    }

    if (converged == 0) std::cerr << "Warning: Bingham closure eigenvalue solver did not converge in " << maxIterations << " iterations with tolerance " << tolerance << '\n';
    return ST;

}