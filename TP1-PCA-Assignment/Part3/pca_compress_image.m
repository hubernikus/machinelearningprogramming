function [A_p, Y, Mu] = pca_compress_image(image_path, Var)
%PCA_COMPRESS_IMAGE Compresses image (PCA) and returns of  A_p, Y and Mu 
% of the last compression in Var

% Set default variables
switch nargin
   case 0
        image_path = '../../TP1-PCA-Dataset/Compression/dream-catchers.mat';
        Var = [0, 0.75, 0.9 , 0.95];        
    case 1
        Var = [0, 0.75, 0.9 , 0.95];
end

% Load image file
load(strcat(image_path));

% Generate Variables
%[M, N]  = size(X);

%% Calculate Eigenvector V, Eigenvalues L and mean Mu
[V, L, Mu] = my_pca(X); 

%% Visuallization
figure('Name', 'Compression of the images','Color', [1 1 1], 'pos', [10 10 1600 750]);

% Setting subplot
rows = ceil(sqrt(max(size(Var))));
cols = ceil(max(size(Var))/rows);

for i = 1:max(size(Var))
    % Choose p to ensure compression given by Var
    p = floor(size(L,1)*(1-Var(i)));
    
    % Projection with pca
    [A_p, Y] = project_pca(X,Mu,V,p);
    
    % Reconstruction of Image (X_hat)
    [X_hat] = reconstruct_pca(Y, A_p, Mu);
        
    % Plot face
    subplot(rows, cols, i); 
    imagesc(X_hat);
    axis off ;
    colormap('Gray');
    
    % Print title
    if Var(i) == 0
        title('Original Image');
    else
        title(['Compressed image by ' num2str(round((Var(i))*100)) '%']);
    end
end

end

