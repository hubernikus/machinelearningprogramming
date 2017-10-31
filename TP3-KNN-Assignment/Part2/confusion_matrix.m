function [C] =  confusion_matrix(y_test, y_est)
%CONFUSION_MATRIX Implementation of confusion matrix 
%   for classification results.
%   input -----------------------------------------------------------------
%
%       o y_test    : (1 x M), a vector with true labels y \in {0,1} 
%                        corresponding to X_test.
%       o y_est     : (1 x M), a vector with estimated labels y \in {0,1} 
%                        corresponding to X_test.
%
%   output ----------------------------------------------------------------
%       o C          : (2 x 2), 2x2 matrix of |TP & FN|
%                                             |FP & TN|.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

true_val = 0;

% True positive (TP)
C(1,1) = sum(and(y_test == true_val, y_est == true_val));

% False positive (FP)
C(2,1) = sum(and(y_test ~= true_val, y_est ==true_val));

% False negatives (FN)
C(1,2) = sum(and(y_test == true_val, y_est ~=true_val));

% False positive (FP)
C(2,2) = sum(and(y_test ~= true_val, y_est ~=true_val));

end

