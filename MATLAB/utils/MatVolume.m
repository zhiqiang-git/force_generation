function [V] = MatVolume(Tet)
%
m = size(Tet.tetFaces, 2);
tets_vol = zeros(1, m);
%
for tId = 1 : m
    p1 = Tet.vertexPoss(1:3, Tet.tetFaces(1, tId));
    p2 = Tet.vertexPoss(1:3, Tet.tetFaces(2, tId));
    p3 = Tet.vertexPoss(1:3, Tet.tetFaces(3, tId));
    p4 = Tet.vertexPoss(1:3, Tet.tetFaces(4, tId));
    e14 = p1 - p4;
    e24 = p2 - p4;
    e34 = p3 - p4;
    tets_vol(tId) = abs(e14'*cross(e24, e34))/6;
end
V = sparse(1:(6*m),1:(6*m), kron(tets_vol, ones(1,6)));