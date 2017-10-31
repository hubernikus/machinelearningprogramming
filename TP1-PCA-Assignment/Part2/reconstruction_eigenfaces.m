function [] = reconstruction_eigenfaces(X, V, Mu, sizeIm)
%RECONSTRUCTION_EIGENFACES Reconstructs face images from lower dimensional
%space to compare the result to the original face image and to the mean face image
%   
n_face = 5; % Number of the face that is compared


% Prepare Figure
fig1 = figure('Name', 'Reconstruction of Face Image','Color', [1 1 1], 'pos', [120 400 1200 200]);

rows = 1;
cols = 6;


% Plot Original Face
    subplot(rows, cols, 6);
    originalFace = reshape(X(:,n_face),sizeIm, sizeIm);
    %colormap = ('Gray');
    imagesc(originalFace);
    colormap('Gray');
    title('Original Face');
    axis('off');

% Plot Mean Face
subplot(rows, cols, 1);
meanFace = reshape(Mu,sizeIm, sizeIm);
imagesc(meanFace);
title('Mean Face');
axis('off');


for p = 1 : 50 : 151    
    
   
    % Project with PCA
    [A_p, Y] = project_pca(X, Mu, V, p);
    
    % Reconstruct Lossy Data from PCA
    [X_hat] = reconstruct_pca(Y, A_p, Mu); % Correct function !!!!
    
    % Reconstructed Faces
    %-> ??
    
    % Plot Reconstructed Faces
    subplot(rows, cols, floor(p/50)+2);
    meanFace = reshape(X_hat(:,n_face),sizeIm, sizeIm);
    imagesc(meanFace);
    title(['p = ' num2str(p)]);
    axis('off');

    
end

end