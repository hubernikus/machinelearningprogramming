function [labels, Mu] =  my_kmeans(X, K, init, type, MaxIter, plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'random','uniform','plus'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters.
%       o Mu       : (N x K), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);
d_i    = zeros(K,M);
k_i    = zeros(1,M);
r_i    = zeros(K,M);
if plot_iter == [];plot_iter = 0;end

% Output Variables
Mu     = zeros(N, K);
labels = zeros(1,M);

%%%%%%%%%%%%%%%%%%%%%%%%% K-Means Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Step 1. Mu Initialization <=== Implement THIS!
Mu = kmeans_init(X, K, init); 

% Visualize Initial Centroids if N=2 and plot_iter active
colors     = hsv(K);
if (N==2 && plot_iter)
    options.title       = sprintf('Initial Mu with <%s> method', init);
    ml_plot_data(X',options); hold on;
    ml_plot_centroid(Mu',colors);
end

iter  = 0;
while true
    Mu_ = Mu;
    
    %%%% Step 2. Distances from X to Mu <=== Implement THIS!
    d_i = my_distX2Mu(X,Mu,type);
    
    
    %%%% Step 3. ssignment Step: Mu Responsability (Eq. 5 and 6) <=== Implement THIS!
    [~,k_i] = min(d_i(:,[1:M]'),[],1);
    
    [k,k_i_matrix] = ndgrid(1:K,k_i);
    r_i = k == k_i_matrix; % K x M - Matrix (X = NxM-Matrix)
    
        
	%%%% Step 4. Update Step: Recompute Mu <=== Implement THIS!
    for jj=1:K
        if sum(r_i(jj,:))
            Mu(:,jj) = (sum(meshgrid(r_i(jj,:),1:N).*X,2))'/sum(r_i(jj,:));
        end
    end
    
    if (N==2 && iter == 1 && plot_iter)
        options.labels      = k_i;
        options.title       = sprintf('Mu and labels after 1st iter');
        ml_plot_data(X',options); hold on;
        ml_plot_centroid(Mu',colors);
    end
    
    %%%% Check for Mu stabilization <=== Implement THIS!    
    if norm(Mu - Mu_) == 0 % Mu same matrix as Mu
        fprintf('Algorithm has converged at iter=%d! Stopping k-means.\n', iter);
        break;
    end
    
    %%%% Check for MaxIter %%%%
    if (iter > MaxIter)
       warning(sprintf('Maximum Niter=%d reached! Stopping k-means.', MaxIter));
       break;
    end
    iter = iter + 1;
end

labels = k_i;

if (N==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroid(Mu',colors);
end

end