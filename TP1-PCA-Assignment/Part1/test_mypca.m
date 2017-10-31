function [ ] = test_mypca( X, V, L , Mu )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[N, M] = size(X);

% Compute PCA with ML_toolbox
options = [];
options.method_name       = 'PCA';
options.nbDimensions      = N;

% Extract Principal Directions, Components and Projection
[~, mappingPCA]  = ml_projection(X',options);

% Test with MLToolbox Results
V_ml = mappingPCA.M;
L_ml = mappingPCA.lambda;
Mu_ml = mappingPCA.mean';

clc;
% Error in Mean Computation
Mu_err = norm(Mu - Mu_ml);
fprintf('[Test1] Error in Mean Computation: %2.2f\n' ,Mu_err);

L_err = norm(diag(L) - L_ml);
fprintf('[Test2] Error in Eigenvalue Computation: %2.2f\n' ,L_err);

V_err = norm(abs(V) - abs(V_ml));
fprintf('[Test3] Error in Eigenvector Computation: %2.2f\n' ,V_err);

end

