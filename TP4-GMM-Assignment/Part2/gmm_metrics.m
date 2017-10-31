function [AIC, BIC] =  gmm_metrics(X, Priors, Mu, Sigma, cov_type)
%GMM_METRICS Computes the metrics (AIC, BIC) for model fitting
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%
%       o AIC      : (1 x 1), Akaike Information Criterion
%       o BIC      : (1 x 1), Bayesian Information Criteria
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Dimensions
[N,M] = size(X);
[K] = length(Priors);
    
% Compute GMM 
logLik = my_gmmLogLik(X,Priors, Mu, Sigma);

% Compute B Parameters
switch cov_type
  case 'full'
    B = K*(1+N+N*(N-1)/2)-1;
  case 'diag'
    B = K*(1+N+N)-1;
  case 'iso'
    B = K*(1+N+1)-1;
  otherwise
    error('Chosen cov_type <<%s>> does not exist', cov_type);
end


% Compute AIC (Equation 13)
AIC = -2*logLik + 2*B;

% Compute AIC (Equation 14)
BIC = -2*logLik + log(M)*B;

end