function [ ll ] = my_gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o ll       : (1 x 1) , loglikelihood
%%
% Assign Dimensions
[N,M] = size(X);
[~,K] = size(Mu);

%Compute the likelihood of each datapoint
pdf = zeros(K,M);
for k = 1:K
    pdf(k,:) = my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));% calculate Probability Density Funciton
end

likelihood = sum(pdf .* ndgrid(Priors,zeros(1,M)),1); % Sum over K

%Compute the total log likelihood
ll = sum(log(likelihood)); % sum of log == log of prod 

end

