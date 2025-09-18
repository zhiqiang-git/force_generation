function [ccs] = ConnectedComponentExtraction(graph, vertexMask)
%
numV = length(vertexMask);
off = 1;
numCCs = 0;
while off <= numV
    if vertexMask(off) == 0
        off = off + 1;
        continue;
    end
    numCCs = numCCs + 1;
    cc = [off];
    vertexMask(off) = 0;
    fringe_start = 0;
    fringe_end = 1;
    while fringe_start < fringe_end
        for i = (fringe_start+1) : fringe_end
            vId = cc(i);
            nIds = graph{vId};
            for j = 1 : length(nIds)
                nId = nIds(j);
                if vertexMask(nId) == 1
                    cc = [cc, nId];
                    vertexMask(nId) = 0;
                end
            end
        end
        fringe_start = fringe_end;
        fringe_end = length(cc);
    end
    ccs{numCCs} = cc;
end