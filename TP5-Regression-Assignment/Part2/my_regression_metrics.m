function [MSE, NMSE, Rsquared] = my_regression_metrics( yest, y )
%MY_REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o Rsquared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Dimensions
[P,M] = size(yest);

% Mean / Average Values
mean_y = 1./M *sum(y,2);
mean_yest = 1./M*sum(yest,2);

% Mean Square Errror (MSE)
MSE = 1./M*sum(sum((yest-y).^2));

% Normaliyed Mean Square Error (NMSE = MSE / VAR(Y)) 
NMSE = MSE/(1./(M-1)*sum(sum((y-repmat(mean_y,1,M)).^2))); %%% Check this function !!

% Coefficient of Determination (R^2)
Rsquared2 = sum(sum((y-repmat(mean_y,1,M)).*(yest-repmat(mean_yest,1,M)),2)).^2 ./ ...
                (sum((y-repmat(mean_y,1,M)).^2,2)).*(sum((yest-repmat(mean_yest,1,M)).^2,2));

Rsquared1 = sum(sum((y-repmat(mean_y,1,M)).*(yest-repmat(mean_yest,1,M)),2)).^2 ./ ...
                 (sum((y-repmat(mean_y,1,M)).^2,2).* sum((yest-repmat(mean_yest,1,M)).^2,2));
        
den = 0;
nom = [0,0];
nom12 = sum((y-repmat(mean_y,1,M)).^2,2);
nom22 = sum((yest-repmat(mean_yest,1,M)).^2,2);
den2 = (sum((y-repmat(mean_y,1,M)).*(yest-repmat(mean_yest,1,M)),2)).^2;
for m = 1:M
    den = den + (y(:,m)-mean_y)*(yest(:,m)-mean_yest);
    nom(1) = nom(1) + (y(:,m)-mean_y)^2;
    nom(2) = nom(2) + (yest(:,m)-mean_yest)^2;
end

%(den^2 == den2)
%(nom(1) == nom12)
%(nom(2) == nom22)
%Rsquared = den^2./(nom(1).*nom(2))


Rsquared = sum(den2./(nom12.*nom22));
%Rsquared1 == Rsquared 
%Rsquared2 == Rsquared


end

