function [matB] = MatGradient(Tet)
%
matG = gradientOperation(Tet);
numV = size(Tet.vertexPoss, 2);
numT = size(Tet.tetFaces, 2);

matG3 =  kron(matG, sparse(eye(3)));
matB = sparse(6*numT, 3*numV);

matB(1:6:(6*numT),:) = matG3(1:9:(9*numT),:);
matB(2:6:(6*numT),:) = matG3(5:9:(9*numT),:);
matB(3:6:(6*numT),:) = matG3(9:9:(9*numT),:);
matB(4:6:(6*numT),:) = (matG3(6:9:(9*numT),:)+matG3(8:9:(9*numT),:))/2;
matB(5:6:(6*numT),:) = (matG3(3:9:(9*numT),:)+matG3(7:9:(9*numT),:))/2;
matB(6:6:(6*numT),:) = (matG3(2:9:(9*numT),:)+matG3(4:9:(9*numT),:))/2;

function [matG] = gradientOperation(Tet)
% Compute undirected neighbors from directed neighbors
numT = size(Tet.tetFaces, 2);
numV = size(Tet.vertexPoss, 2);
rowsG = ones(4,1)*(1:(3*numT));
colsG = kron(Tet.tetFaces, ones(1,3));
valsG = zeros(4, 3*numT);
%
for tId = 1 : numT
    valsG(:,(3*tId-2):(3*tId)) = tetGradient(Tet.vertexPoss(1:3,...
        Tet.tetFaces(:,tId)))';
end
matG = sparse(rowsG, colsG, valsG, 3*numT, numV);

function [g3x4] = tetGradient(vPoss)
% \|[ones(1,4); vPoss]'*(a;b) - f\|^2
TP = [ones(1,4); vPoss]';
A = inv(TP'*TP)*TP';
g3x4 = A(2:4, :);