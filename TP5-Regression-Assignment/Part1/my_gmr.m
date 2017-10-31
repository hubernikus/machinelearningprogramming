function [y_est, var_est] = my_gmr(Priors, Mu, Sigma, X, in, out)
%MY_GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o x:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs. ???? N-> M
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check dimensions of x
if(size(X,1) ~= length(in))
    if(size(X,2) == length(in))
        warning('X has the wrong format. Matrix X is beeing transposed.')
        X = X';
    else 
        error('X has the wrong format')
    end
end

% Initialize variables
[D, K] = size(Mu); % K- Number of Clusters; Dimensions total
[N, M] = size(X); % M - Datapoints; N - Dimeions input

P = D-N; % Dimensionality output

% Reproduce Sigma, Mu Priors sorted
Mu_sort = zeros(D,K); Sigma_sort = zeros(D,D,K);
ind_tot = [in, out];
for ii = 1:D
    Mu_sort(ii,:) = Mu(ind_tot(ii),:);
    for jj= 1:D
        Sigma_sort(ii,jj,:) = Sigma(ind_tot(ii),ind_tot(jj),:);     
    end
end         

beta_nom = zeros(K,M); % Nominator of mixing weight
for k = 1:K
    probPDF = my_gaussPDF(X, Mu_sort(1:N,k), Sigma_sort(1:N,1:N,k));
    beta_nom(k,:) = Priors(k)*probPDF;    
end

beta_k = beta_nom./repmat(sum(beta_nom,1),K,1); % Normalize mixing weight

% Local regressive function
Mu_tild = zeros(P,M,K);
for k = 1:K
    Mu_tild(:,:,k) = repmat(Mu_sort(N+1:end,k),1,M) + Sigma_sort(N+1:end,1:N,k) ...
        *pinv(Sigma_sort(1:N,1:N,k))*(X-repmat(Mu_sort(1:N,k), 1, M));
end

% Regressive function obatined by computing expectaion
y_est = zeros(P,M);
for k = 1:K
    y_est  = y_est + repmat(beta_k(k,:),P,1) .* Mu_tild(:,:,k);
end

% Conditional Density for each regressive function
Sigma_tild = zeros(P,P,K);
for k = 1:K
    Sigma_tild(:,:,k) =  Sigma_sort(N+1:end,N+1:end,k) - Sigma_sort(N+1:end,1:N,k)...
        /(Sigma_sort(1:N,1:N,k))*Sigma_sort(1:N,N+1:end,k);
end

% Caluclate Sum
var_est = zeros(P,P,M);
for m = 1:M
    var_estSum = zeros(P); % Last therm has to be summed up before it's squared
    for k = 1:K
        var_est(:,:,m) = var_est(:,:,m) + beta_k(k,m)*(Mu_tild(:,m,k)*Mu_tild(:,m,k)' +Sigma_tild(:,:,k));
        var_estSum = var_estSum +beta_k(k,m)*Mu_tild(:,m,k);        
    end
    var_est(:,:,m) = var_est(:,:,m) - var_estSum * var_estSum';
end