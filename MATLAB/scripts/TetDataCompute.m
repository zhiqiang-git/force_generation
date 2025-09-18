function [Tet_out] = TetDataCompute(Tet)
% The goal is to compute neighoring vertices and boundary vertices
% Compute adjacent vertices of each vertex. We only store an edge [sId, tId]
% in neighbors{sId} if sId < tId
neighbors = getEdges(Tet.tetFaces);
% Compute traingles of the tet mesh. Each triangle is given by [v1Id, v2Id,
% v3Id]. It is only stored in triangles{v1Id}. triangles{v1Id} is a 6xn
% matrix, where n is the number of adjacent traingles of vertex 'v1Id'.
% Each column stores [v1Id,v2Id, v3Id, #adjacentTets,
% adjacentTet1Id, adjacentTet2Id]. if a triangle is a boundary triangle,
% then adjacentTet2Id = 0
% Initialize the triangle data structure
triangles = getTriangles(neighbors); 
% extract boundary triangles and determine the adjacent tets of each
% trianlge
triangles = detectBoundaryTriangles(Tet.tetFaces, triangles); 
Tet_out = Tet;
% determine the boundary vertices
[IVFlags, Tet_out.boundaryVertexIndices] = extractBoundaryVertices(Tet.tetFaces, triangles);
Tet_out.vertexPoss = [Tet_out.vertexPoss(1:3,:); IVFlags];
Tet_out.neighbors = neighbors;
Tet_out.triangles = triangles;

% Compute boundary faces and vertex normals. For interior vertices, their 
% normals are [0,0,0]' 
[Tet_out.boundaryFaceVIds, Tet_out.boundaryFaceTIds, Tet_out.vertexNors] =...
    BoundaryVertexNormal(Tet_out);
% Compute boudary face areas
Tet_out.boundaryVertexArea = BoundaryVertexArea(Tet_out.vertexPoss,...
    Tet_out.boundaryFaceVIds);


%
function [vertexArea] = BoundaryVertexArea(vertexPoss, triangles)
%
numV = size(vertexPoss, 2);
vertexArea = zeros(1, numV);
numT = size(triangles, 2);
for tId = 1 : numT
    v1Id = triangles(1, tId);
    v2Id = triangles(2, tId);
    v3Id = triangles(3, tId);
    p1 = vertexPoss(:, v1Id);
    p2 = vertexPoss(:, v2Id);
    p3 = vertexPoss(:, v3Id);
    e12 = p1 - p2;
    e13 = p1 - p3;
    e23 = p2 - p3;
    a = e12'*e12;
    b = e13'*e13;
    c = e23'*e23;
    area = sqrt(2*(a*b+a*c+b*c)-a^2-b^2-c^2)/4;
    vertexArea(v1Id) = vertexArea(v1Id) + area/3;
    vertexArea(v2Id) = vertexArea(v2Id) + area/3;
    vertexArea(v3Id) = vertexArea(v3Id) + area/3;
end

% function [graph_bv] = boundaryVertexGraph(bVInds, bTris)
% numV = max(max(bTris));
% rows = [bTris(1,:), bTris(2,:), bTris(3,:)];
% cols = [bTris(2,:), bTris(3,:), bTris(1,:)];
% A = sparse(rows, cols, ones(1,length(rows)), numV, numV);
% A = A + A';
% A = A(bVInds, bVInds);
% numBV = length(bVInds);
% for id = 1 : numBV
%     graph_bv{id} = find(A(id,:));
% end

function [IVFlags, BVIds] = extractBoundaryVertices(tetFaces, triangles)
numV = length(triangles);
IVFlags = ones(1, numV);
for vId = 1 : numV
    tris = triangles{vId};
    for i = 1 : size(tris, 2)
        if tris(4,i) == 1
            IVFlags(tris(1:3,i)) = 0;
        end
    end
end
BVIds = find(IVFlags == 0);

%
function [triangles_out] = detectBoundaryTriangles(tetFaces, triangles)
%
nT = length(tetFaces);
for tId = 1 : nT
    vids = sort(tetFaces(:,tId));
    tris1 = triangles{vids(1)};
    off = find(tris1(2,:) == vids(2) & tris1(3,:) == vids(3));
    triangles{vids(1)}(4, off) =  triangles{vids(1)}(4, off) + 1;
    if triangles{vids(1)}(4, off) == 1
        triangles{vids(1)}(5, off) =  tId;
    else
        triangles{vids(1)}(6, off) =  tId;
    end
    off = find(tris1(2,:) == vids(2) & tris1(3,:) == vids(4));
    triangles{vids(1)}(4, off) =  triangles{vids(1)}(4, off) + 1;
    if triangles{vids(1)}(4, off) == 1
        triangles{vids(1)}(5, off) =  tId;
    else
        triangles{vids(1)}(6, off) =  tId;
    end
    off = find(tris1(2,:) == vids(3) & tris1(3,:) == vids(4));
    triangles{vids(1)}(4, off) =  triangles{vids(1)}(4, off) + 1;
    if triangles{vids(1)}(4, off) == 1
        triangles{vids(1)}(5, off) =  tId;
    else
        triangles{vids(1)}(6, off) =  tId;
    end
    tris2 = triangles{vids(2)};
    off = find(tris2(2,:) == vids(3) & tris2(3,:) == vids(4));
    triangles{vids(2)}(4, off) =  triangles{vids(2)}(4, off) + 1;
    if triangles{vids(2)}(4, off) == 1
        triangles{vids(2)}(5, off) =  tId;
    else
        triangles{vids(2)}(6, off) =  tId;
    end
end
triangles_out = triangles;

function [triangles] = getTriangles(neighbors)
%
numV = length(neighbors);
%
triangles = cell(1, numV);
for vId = 1 : numV
    nIds = sort(neighbors{vId});
    buf = [];
    for j = 1 : length(nIds)
        v1Id = nIds(j);
        n1Ids = neighbors{v1Id};
        for k = (j+1): length(nIds)
            v2Id = nIds(k);
            if length(find(n1Ids == v2Id)) > 0
                buf = [buf,[vId, v1Id, v2Id,0,0,0]'];
            end
        end
    end
    triangles{vId} = buf;
end

function [neighbors] = getEdges(tetFaces)
%
numV = max(max(tetFaces));
numT = size(tetFaces, 2);
v1Ids = tetFaces(1,:);
v2Ids = tetFaces(2,:);
v3Ids = tetFaces(3,:);
v4Ids = tetFaces(4,:);
%
rows = [v1Ids, v1Ids, v1Ids, v2Ids, v2Ids, v3Ids];
cols = [v2Ids, v3Ids, v4Ids, v3Ids, v4Ids, v4Ids];
vals = ones(1, 6*numT);
A = sparse(rows,cols, vals, numV, numV);
A = A + A';
[rows, cols, ~] = find(A);
neighbors = cell(1, numV);
for tId = 1 : length(rows)
    if rows(tId) > cols(tId)
        v1Id = cols(tId);
        v2Id = rows(tId);
        neighbors{v1Id} = [neighbors{v1Id}, v2Id];
    end
end
