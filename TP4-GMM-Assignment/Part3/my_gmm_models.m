function [models] = my_gmm_models(X_train, y_train, K, cov_type, plot_iter)
%MY_GMM_MODELS Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters for each class of
% the dataset X_train
%   input------------------------------------------------------------------
%
%       o X_train   : (N x M_train), a data set with M_train samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o y_train   : (1 x M_train), a vector with labels y corresponding to X_train.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o plot_iter : (bool)  set to 1 of want to visalize initual Mu's and
%                          Sigma's, works only for N=2
%
%   output ----------------------------------------------------------------
%       o models    :  (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
% Define Dimensions
[N, M_train] =size(X_train);

% Initialization
labels = unique(y_train);
N_classes = length(labels);

% Initialize matrizes
Priors = zeros(1,K); Mu = zeros(N,K); Sigma = zeros(N,N,K);
models = repmat(struct('Priors',Priors,'Mu',Mu, 'Sigma', Sigma),[1,N_classes]);


%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
for c = 1:N_classes
    [Priors, Mu, Sigma] = my_gmmEM(X_train(:,y_train == labels(c)), K, cov_type, plot_iter);

    % Build GMM for the current class and create model struct
    models(c) = struct('Priors',Priors,'Mu',Mu, 'Sigma', Sigma);
end


end