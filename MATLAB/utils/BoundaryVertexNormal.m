function [boundaryFaceVIds,boundaryFaceTIds,vertexNors] = BoundaryVertexNormal(Tet)
%
numV = size(Tet.vertexPoss, 2);
numBT = 0;
for vId = 1:numV
    tris = Tet.triangles{vId};
    for i = 1 : size(tris, 2)
        if tris(4, i) == 1
            numBT = numBT + 1;
        end
    end
end
boundaryFaceVIds = zeros(3, numBT);
boundaryFaceTIds = zeros(1, numBT);
triId = 0;
vertexNors = zeros(3, numV);
for vId = 1:numV
    tris = Tet.triangles{vId};
    for i = 1 : size(tris, 2)
        if tris(4, i) == 1
            tId = tris(5,i);
            v4id = sum(Tet.tetFaces(:,tId)) - sum(tris(1:3,i));
            p1 = Tet.vertexPoss(1:3, tris(1, i));
            p2 = Tet.vertexPoss(1:3, tris(2, i));
            p3 = Tet.vertexPoss(1:3, tris(3, i));
            p_interior = Tet.vertexPoss(1:3, v4id);
            [fNor, flip] = tet_face_norml(p1,p2,p3, p_interior);
            triId = triId + 1;
            if flip == 1
                boundaryFaceVIds(:, triId) = tris([1,3,2], i);
                boundaryFaceTIds(:, triId) = tris(5, i);
            else
                boundaryFaceVIds(:, triId) = tris([1,2,3], i);
                boundaryFaceTIds(:, triId) = tris(5, i);
            end
            vertexNors(:, tris(1,i)) = vertexNors(:, tris(1,i)) + fNor;
            vertexNors(:, tris(2,i)) = vertexNors(:, tris(2,i)) + fNor;
            vertexNors(:, tris(3,i)) = vertexNors(:, tris(3,i)) + fNor;
        end
    end
end
for vId = 1:numV
    len = norm(vertexNors(:,vId));
    if len > 0.1
        vertexNors(:,vId) = vertexNors(:,vId)/len;
    end
end

function [fNor, flip] = tet_face_norml(p1,p2,p3, p_interior)
% Compute the face normal of the face whose boundary vertices are p1, p2,
% p3. We want to use the interior vertex position given by p_interior
%
e12 = p1 - p2;
e13 = p1 - p3;
fNor = cross(e12, e13);
fNor = fNor/norm(fNor);
sign = (p_interior - p1)'*fNor;
if sign > 0
    flip = 1;
    fNor = -fNor;
else
    flip = 0;
end