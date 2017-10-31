function [  Priors, Mu, Sigma ] = my_gmmEM(X, K, cov_type, plot_iter)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o plot_iter : (bool)  set to 1 of want to visalize initual Mu's and
%                          Sigma's, works only for N=2
%       o verb      : (bool)  set to 1 of want to see the convergence output
%   output ----------------------------------------------------------------
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%

% Define variables
[N, M] = size(X);

% Initialize Matrizes
prob_PDF = zeros(K,M);

if nargin<4 plot_iter=0; end %Set default value for plot_iter
[Priors, Mu, Sigma] = my_gmmInit(X, K, cov_type, plot_iter);
close all;

% Variables for maximum iteration
iter = 1; maxIter = 300;

logLik = my_gmmLogLik(X, Priors, Mu, Sigma);

while true

    %%%%%% STEP 2: Expectation Step: Membership probabilities %%%%%%
    
    % 1) Compute probabilities p(x^i|k)
    for k = 1:K
        prob_PDF(k,:) = my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
    end
    % 2) Compute posterior probabilities p(k|x)  %%%
    probPDF_sum = sum(ndgrid(Priors, zeros(M,1)).*prob_PDF,1);
        
    prob_aPost = (ndgrid(Priors,zeros(1,M)).*prob_PDF)./meshgrid(probPDF_sum,zeros(1,K));
            
    %%%%%% STEP 3: Maximization Step: Update Priors, Means and Sigmas %%%%%%    

    % 1) Update Priors
    Priors = 1./M*sum(prob_aPost,2)';
    
    % 2) Update Means and Covariance Matrix
    
    Mu = zeros(N,K);
    Sigma = zeros(N,N,K);
    for k = 1:K
        % Update Means
        Mu(:,k) = sum(meshgrid(prob_aPost(k,:),zeros(1,N)).*X,2)/sum(prob_aPost(k,:));
    
        % Update Covariance Matrices 
        X_Mu = X - ndgrid(Mu(:,k), zeros(M,1));
        switch cov_type
          case 'iso'
            for m = 1:M
                Sigma(:,:,k) = Sigma(:,:,k) + 1./N*prob_aPost(k,m)* ...
                    (X(:,m)-Mu(:,k))'*(X(:,m)-Mu(:,k))*diag(ones(N,1));
            end
            Sigma(:,:,k) = Sigma(:,:,k)/sum(prob_aPost(k,:));
             %Sigma(:,:,k) = Sigma(:,:,k) + 1./N*prob_aPost(k,m)* ...
             %       (X(:,m)-Mu(:,k))'*(X(:,m)-Mu(:,k))*diag(ones(N,1));
          case 'full'
            Sigma(:,:,k) = (meshgrid(prob_aPost(k,:),zeros(N,1)).*X_Mu)*X_Mu' ...
                /sum(prob_aPost(k,:)); 
          case 'diag'
            Sigma(:,:,k) = (meshgrid(prob_aPost(k,:),zeros(N,1)).*X_Mu)*X_Mu' ...
                /sum(prob_aPost(k,:)); 
            Sigma(:,:,k) = diag(diag(Sigma(:,:,k)));
          otherwise      
            error('Covariance Type %s does not exist', cov_tpe);
        end
        
        % Add a tiny variance to avoid numerical instability
        Sigma(:,:,k) = Sigma(:,:,k) + 1e-5*diag(ones(N,1));
    end    
    
    %%%%%% Stopping criterion %%%%%%
    logLik_ = logLik;
    logLik = my_gmmLogLik(X, Priors, Mu, Sigma);
     
    if abs(logLik -logLik_) < 1e-6
        fprintf('Algorithm has converged, iter=%d. Stopping gmmEM. \n', iter);
        break;
    end
   
   %%%%% Check for MaxIter %%%%%%%
    if (iter > maxIter)
        warning('Maximu m Niter=%d reached! Stopping gmmEM. \n', ...
                maxIter');
        break;
    end
    iter = iter + 1;
end

%%%%%% Visualize Final Estimates %%%%%%
if (N==2 && plot_iter==1)
options.labels      = [];
options.class_names = {};
options.plot_figure = false;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu',colors);hold on; 
plot_gmm_contour(gca,Priors,Mu,Sigma,colors);
title('Final GMM Parameters');
grid on; box on;

end
