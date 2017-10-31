    function [RSS, AIC, BIC] =  my_metrics(X, labels, Mu)
%MY_METRICS Computes the metrics (RSS, AIC, BIC) for clustering evaluation
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%
%   output ----------------------------------------------------------------
%
%       o RSS      : (1 x 1), Residual Sum of Squares
%       o AIC      : (1 x 1), Akaike Information Criterion
%       o BIC      : (1 x 1), Bayesian Information Criteria
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary Variables
[D, M] = size(X);
[~, K] = size(Mu);

% Compute RSS (Equation 8)
%RSS = inf;    
RSS = 0;
for m = 1:M
    deltaRSS = X(:,m) - Mu(:,labels(m));
    RSS = RSS + deltaRSS'*deltaRSS ;
end



% Compute AIC (Equation 9)
%AIC = inf;
AIC = RSS + 2*K*D;


% Compute AIC (Equation 10)
%BIC = inf;
BIC = RSS + log(M)*K*D;

end