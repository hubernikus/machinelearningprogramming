function [labels] =  gmm_cluster(X, Priors, Mu, Sigma, type, softThresholds)
%GMM_CLUSTER Computes the cluster labels for the data points given the GMM
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                           Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o type   : string ,{'hard', 'soft'} type of clustering
%
%       o softThresholds: (2 x 1), a vecor for the minimum and maximum of
%                           the threshold for soft clustering in that order
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a M dimensional vector with the label of the
%                             cluster for each datapoint
%                             - For hard clustering, the label is the 
%                             cluster number.
%                             - For soft clustering, the label is 0 for 
%                             data points which do not have high confidnce 
%                             in cluster assignment
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define dimensions
[N,M] = size(X);
K = length(Priors);

% Initialize Matrizes
prob_PDF = zeros(K,M);

% Find the a posteriori probability for each data point for each cluster
% 1) Compute probabilities p(x^i|k)
for k = 1:K
    prob_PDF(k,:) = my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
end
% 2) Compute posterior probabilities p(k|x)  %%%
probPDF_sum = sum(ndgrid(Priors, zeros(M,1)).*prob_PDF,1);
        
prob_aPost = (ndgrid(Priors,zeros(1,M)).*prob_PDF)./meshgrid(probPDF_sum,zeros(1,K));

% Use posterior probabilities to assign points to clusters based on
% clustering method 'hard' or 'soft'
%for ii = 1:M

    switch type
        case 'hard'
            % Find the cluster with highest probability
            [~,labels] = max(prob_aPost,[],1);
                
        case 'soft'
            % Find the cluster with highest probabilty. Unless, the highest
            % and another cluster are in the same range specified by
            % threshold
            
            % Find cluster with highest probability
            [max_prob,labels] = max(prob_aPost,[],1);
            
            % --- Check confidence
            % Check if the Maximum is outside range > True
            highMax = max_prob > softThresholds(2);
            
            % Check if all the other data points are outside the range
            prob_aPost(labels + ((1:M)-1)*K) = softThresholds(1); %Assign Maximum equal to t_min
            lowRest = not(sum(prob_aPost > softThresholds(1),1));
                        
            % Assign to zero datapoints with having both conditions 
            labels = labels .* (highMax | lowRest);

        otherwise
            fprintf('Invalid type for clustering\n');
    end
%end

