function [OPT] = WCSA_Prepare(Tet, Basis, OPT_in, eps, M_r)
%
OPT = OPT_in;
% Set the tets weights for each eigenmode
numEigenModes = length(Basis.eigenVals);
numTets = size(Tet.tetFaces, 2);
OPT.tetWeights = zeros(numEigenModes, numTets);
for id = 1 : numEigenModes
    OPT.tetWeights(id,:) = StressProcessing(OPT, Basis.eigenVecs(:,id), 2);
end
% Compute the weak regions
tetAdjGraph = extractTetAdjGraph(Tet);
numActiveTets = floor(eps*numTets);
for id = 1 : numEigenModes
    [~, ids] = sort(OPT.tetWeights(id,:), 'descend');
    activeTets = ids(1:numActiveTets);
    mask = zeros(1, numTets);
    mask(activeTets) = 1;
    ccs = ConnectedComponentExtraction(tetAdjGraph, mask);

    % Compute the scores of each cluster, keep the top M_r
    ccs_scores = zeros(1, length(ccs));
    for i = 1:length(ccs)
        ccs_scores(i) = max(OPT.tetWeights(id, ccs{i}));
    end
    [ccs_scores, ids] = sort(ccs_scores, 'descend');
    % length(ids) is usually around 100, larger than M_r=5
    ids = ids(1:min(length(ids), M_r));
    ccs_scores = ccs_scores(1:length(ids));
    for i = 1 : length(ids)
        % OPT.weakRegions{id}{i} = extractActiveBoundaryVertices(Tet, ccs{ids(i)});
        OPT.weakRegions{id}{i} = extractActiveVertices(Tet, ccs{ids(i)});
        OPT.weakRegionTets{id}{i} = ccs{ids(i)};
    end
end
%

%
OPT.L = OPT.B'*OPT.V*OPT.D*OPT.B;
numV = size(Tet.vertexPoss, 2);
numBV = length(Tet.boundaryVertexIndices);
rows = 3*kron(Tet.boundaryVertexIndices, ones(1,3)) + kron(ones(1, numBV), [-2,-1,0]);
cols = kron(1:numBV, ones(1,3));
vals = reshape(Tet.vertexNors(:, Tet.boundaryVertexIndices), [1,3*numBV]);
OPT.N = sparse(rows, cols, vals, 3*numV, numBV);
OPT.A = sparse(1:numBV, 1:numBV, Tet.boundaryVertexArea(Tet.boundaryVertexIndices));
OPT.mat_NA = OPT.N*OPT.A;
OPT.mat_SigmaNA = zeros(3, numBV);
OPT.mat_SigmaNA(1,:) = sum(OPT.mat_NA(1:3:(3*numV),:));
OPT.mat_SigmaNA(2,:) = sum(OPT.mat_NA(2:3:(3*numV),:));
OPT.mat_SigmaNA(3,:) = sum(OPT.mat_NA(3:3:(3*numV),:));
OPT.mat_SigmaTNA = zeros(3, numBV);
for vId = 1 : numV
    pos = Tet.vertexPoss(:, vId);
    OPT.mat_SigmaTNA(1,:) = OPT.mat_SigmaTNA(1,:)...
        + pos(2)*OPT.mat_NA(3*vId,:) - pos(3)*OPT.mat_NA(3*vId-1,:);
    OPT.mat_SigmaTNA(2,:) = OPT.mat_SigmaTNA(2,:)...
        + pos(3)*OPT.mat_NA(3*vId-2,:) - pos(1)*OPT.mat_NA(3*vId,:);
    OPT.mat_SigmaTNA(3,:) = OPT.mat_SigmaTNA(3,:)...
        + pos(1)*OPT.mat_NA(3*vId-1,:) - pos(2)*OPT.mat_NA(3*vId-2,:);
end
OPT.mat_Sigma = kron(ones(1, numV), eye(3));
OPT.mat_SigmaT = zeros(3, 3*numV);
OPT.mat_SigmaT(1, 2:3:(3*numV)) = -Tet.vertexPoss(3,:);
OPT.mat_SigmaT(1, 3:3:(3*numV)) = Tet.vertexPoss(2,:);
OPT.mat_SigmaT(2, 1:3:(3*numV)) = Tet.vertexPoss(3,:);
OPT.mat_SigmaT(2, 3:3:(3*numV)) = -Tet.vertexPoss(1,:);
OPT.mat_SigmaT(3, 1:3:(3*numV)) = -Tet.vertexPoss(2,:);
OPT.mat_SigmaT(3, 2:3:(3*numV)) = Tet.vertexPoss(1,:);

% Extract active boundary vertices that correspond to active tets
function [activeBVIds] = extractActiveBoundaryVertices(Tet, activeTetIds)
numT = size(Tet.tetFaces, 2);
activeTetFlags = zeros(1, numT);
activeTetFlags(activeTetIds) = 1;
numBV = length(Tet.boundaryVertexIndices);
activeBVIds = zeros(1, numBV);
for i = 1 : numBV
    bVId = Tet.boundaryVertexIndices(i);
    tris = Tet.triangles{bVId};
    for j = 1 :size(tris, 2)
        if tris(4, j) == 1
            if activeTetFlags(tris(5,j)) == 1
                % 3.
                % Zhiqiang: Problem is one triangle is store in only
                % one vertex's information, may miss some vertices
                activeBVIds(i) = 1;
                break;
            end
        end
    end
end
activeBVIds = find(activeBVIds);

function [activeVIds] = extractActiveVertices(Tet, activeTetIds)
numT = size(Tet.tetFaces, 2);
activeTetFlags = zeros(1, numT);
activeTetFlags(activeTetIds) = 1;
numV = size(Tet.vertexPoss, 2);
activeVIds = zeros(1, numV);
for VId = 1 : numV
    tris = Tet.triangles{VId};
    if size(tris, 2) > 0
        for j = 1 :size(tris, 2)
            if tris(4, j) == 2
                if activeTetFlags(tris(5, j)) == 1 || activeTetFlags(tris(6, j)) == 1
                    activeVIds(VId) = 1;
                    activeVIds(tris(2, j)) = 1;
                    activeVIds(tris(3, j)) = 1;
                    break
                end
            elseif tris(4, j) == 1
                if activeTetFlags(tris(5, j)) == 1
                    activeVIds(VId) = 1;
                    activeVIds(tris(2, j)) = 1;
                    activeVIds(tris(3, j)) = 1;
                    break
                end
            end
        end
    end
end
activeVIds = find(activeVIds);

function [tetAdjGraph] = extractTetAdjGraph(Tet)

numInteriorTriangles = 0;
numV = length(Tet.triangles);
for vId = 1 : numV
    tris = Tet.triangles{vId};
    if size(tris, 2) > 0
        numInteriorTriangles = numInteriorTriangles + length(find(tris(4,:)==2));
    end
end
rows = zeros(1, numInteriorTriangles);
cols = zeros(1, numInteriorTriangles);
vals = ones(1, numInteriorTriangles);
off = 0;
for vId = 1 : numV
    tris = Tet.triangles{vId};
    if size(tris, 2) > 0
        tp = find(tris(4,:) == 2);
        num = length(tp);
        rows((off+1):(off+num)) = tris(5,tp);
        cols((off+1):(off+num)) = tris(6,tp);
        off = off + num;
    end
end
numT = size(Tet.tetFaces, 2);
adjGraphMat = sparse(rows, cols, vals, numT, numT);
adjGraphMat = adjGraphMat + adjGraphMat';
tetAdjGraph = cell(1, numT);
[rows, cols, ~] = find(adjGraphMat);
tId = 1;
startId = 1;
for id = 1 : length(rows)
    if cols(id) ~= tId
        tetAdjGraph{tId} = rows(startId:(id-1));
        tId = cols(id);
        startId = id;
    end
end

