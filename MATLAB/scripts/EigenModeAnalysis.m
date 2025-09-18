function [OPT, Basis] = EigenModeAnalysis(Tet, type, numEigens)
% The matrices derived from model analysis are stored in OPT,
% i.e., OPT.B, OPT.V, and OPT.D 
OPT.V = MatVolume(Tet);
matC = MatElasticity(type);
numT = size(Tet.tetFaces, 2);
OPT.D = kron(sparse(1:numT, 1:numT, ones(1, numT)), matC);
OPT.B = MatGradient(Tet);
L = OPT.B'*OPT.V*OPT.D*OPT.B;
% The first six eigenvalues are zero
[eigenVecs, eigenVals] = eigs(L, numEigens + 6, -1e-10);
eigenVals = diag(eigenVals)';
% This is usually application specific
for id = 1 : numEigens
    Basis.eigenVals(id) = eigenVals(id+6);
    vertexNorms = Tet.vertexNors(:);
    f_ext = eigenVals(id+6)*eigenVecs(:, id+6);
    metric = sum(vertexNorms.*f_ext);
    if metric <= 0
        Basis.eigenVecs(:,id) = eigenVecs(:, id+6);
    else 
        Basis.eigenVecs(:,id) =-eigenVecs(:, id+6);
    end
end