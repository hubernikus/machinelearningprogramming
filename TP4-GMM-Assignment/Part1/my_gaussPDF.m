function prob = my_gaussPDF(X, Mu, Sigma)
%MY_GAUSSPDF computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by a mean and covariance matrix.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 matrix corresponding to the centroid mu_k \in R^Nn
%       o Sigma : (N x N), an NxN matric representing the covariance matrix of the 
%                          Gaussian function
% Outputs ----------------------------------------------------------------
%       o prob  : (1 x M),  a 1xM vector representing the probabilities for each 
%                           M datapoints given Mu and Sigma    
%%
% Define Dimenions
[N, M] = size (X);

% Initalize Matrizes
prob = zeros(1,M);

prob_frac = 1./((2*pi)^(N/2)*det(Sigma)^(0.5)); % Claculate the fraction outside of the loop
inv_Sigma = pinv(Sigma); % Calculate inverse outside of the loop to speed up
for m = 1:M
    prob(m) = prob_frac * exp(-0.5*(X(:,m)-Mu)'*inv_Sigma*(X(:,m)-Mu)); %use pinv to avoid singularities
end