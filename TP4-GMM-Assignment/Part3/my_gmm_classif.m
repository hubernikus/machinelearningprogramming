function [y_est] = my_gmm_classif(X_test, models, labels, K, P_class)
%MY_GMM_CLASSIF Classifies datapoints of X_test using ML Discriminant Rule
%   input------------------------------------------------------------------
%
%       o X_test    : (N x M_test), a data set with M_test samples each being of
%                           dimension N, each column corresponds to a datapoint.
%       o models    : (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o labels    : (1 x N_classes) unique labels of X_test.
%       o K         : (1 x 1) number K of GMM components.
%   optional---------------------------------------------------------------
%       o P_class   : (1 x N_classes), the vector of prior probabilities
%                      for each class i, p(y=i). If provided, equal class
%                      distribution assumption is no longer made.
%
%   output ----------------------------------------------------------------
%       o y_est  :  (1 x M_test), a vector with estimated labels y \in {0,...,N_classes}
%                   corresponding to X_test.
%%

% Define Dimensions
[~, M_test] = size(X_test);
N_classes = length(labels);

% Create variables 
sum_probPDF = zeros(N_classes, M_test);

% Calculate Sum of Priors * Probability (PDF)  over K [CxM_test]
for c = 1:N_classes
    for k = 1:K
        sum_probPDF(c,:) = sum_probPDF(c,:) +   ...
            models(c).Priors(k)*my_gaussPDF(X_test, models(c).Mu(:,k), models(c).Sigma(:,:,k));
    end
end

% Calculate the most probable class for each datapoint 
switch nargin
    case 4 % Assuming equal class distribution
        [~,ind_minLog] = min(-log(sum_probPDF),[],1);
    case 5 % Distribution given by P_class
        [~,ind_minLog] = min(-log(sum_probPDF.*ndgrid(P_class,zeros(1,M_test))),[],1);
        
    otherwise
        warning('Wrong number of arguments for my_gmm_classif! \n');
end

% Assign the corresponding class-label
y_est = labels(ind_minLog);