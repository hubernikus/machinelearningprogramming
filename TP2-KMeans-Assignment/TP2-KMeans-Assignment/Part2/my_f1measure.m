function [f1measure] =  my_f1measure(test_labels, true_labels)
%MY_FMEASURE Computes the f1-measure between two labels for a dataset (as column vectors)
%   depending on the choosen distance type={'L1','L2','LInf'}
%
%   input -----------------------------------------------------------------
%   
%       o true_labels     : (M x 1),  M-dimensional vector with true labels for
%                                     each data point
%       o true_labels     : (M x 1),  M-dimensional vector with classified labels for
%                                     each data point
%   output ----------------------------------------------------------------
%
%       o f1_measure      : f1-measure for the classified labels
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(true_labels);
true_K = unique(true_labels);
found_K = unique(test_labels);

nClasses = length(true_K);
nClusters = length(found_K);

% Initializing variables

P = zeros(nClusters, nClasses);
R = zeros(nClusters, nClasses);
F1 = zeros(nClusters, nClasses);

for ii = 1:nClasses
    for jj = 1:nClusters
        
        n_ik = 0;
        for m = 1:M
            if and(	true_labels(m) == true_K(ii), test_labels(m) == found_K(jj))
                n_ik = n_ik + 1;
            end
        end
        
% Implement the precision equation here
        P(ii, jj) = n_ik / sum(test_labels == found_K(jj));
        
% Implement the recall equation here
        % True labels, divised by found labels in class
        R(ii, jj) = n_ik / sum(true_labels == true_K(ii));

% Implement the F1 measure for each cluster here
        F1(ii,jj) = 2*P(ii,jj)*R(ii,jj)/(R(ii,jj)+P(ii,jj));
        
    end
end


% Implement the F1 measure for all clusters here
%f1measure = sum(sum(true_labels == true_K(1:nClasses))/m*max(F1,[],1));
%f1measure = sum(sum(true_labels == true_K(1:nClasses))/m * max(F1(1:nClasses,:)));
f1measure = 0;
for ii = 1:nClasses
    f1measure = sum(true_labels == true_K(ii))/M * max(F1(ii,:)) + f1measure;
end

end
