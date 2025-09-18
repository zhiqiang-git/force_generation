function [Tet] = LoadTetMesh(filename)
%
numPoss_alloc = 16384;
numTets_alloc = 32768*2;
Tet.vertexPoss = zeros(3, numPoss_alloc);
Tet.tetFaces = zeros(4, numTets_alloc);
numPoss = 0;
numTets = 0;
flag_load_vPoss = 0;
flag_load_tets = 0;
f_id = fopen(filename, 'r');
while 1
    tline = fgetl(f_id);
    if tline == -1
        break;
    end
    if flag_load_vPoss == 1 && tline(1) == ']'
        flag_load_vPoss = 0;
        flag_load_tets = 0;
        continue;
    end
    if flag_load_tets == 1 && tline(1) == ']'
        flag_load_vPoss = 0;
        flag_load_tets = 0;
        continue;
    end
    if length(tline)>= 7 && strcmp(tline(1:7),'msh.POS') == 1
        flag_load_vPoss = 1;
        flag_load_tets = 0;
        continue;
    end
    if length(tline)>= 8 && strcmp(tline(1:8),'msh.TETS') == 1
        flag_load_vPoss = 0;
        flag_load_tets = 1;
        continue;
    end
    if flag_load_vPoss == 0 && flag_load_tets == 0
        continue;
    end
    if flag_load_vPoss == 1
        pos = str2num(tline(1:(length(tline)-1)));
        numPoss = numPoss + 1;
        if numPoss > numPoss_alloc
            Tet.vertexPoss = [Tet.vertexPoss, zeros(3,numPoss_alloc)];
            numPoss_alloc = numPoss_alloc*2;
        end
        Tet.vertexPoss(:, numPoss) = pos';
    end
    if flag_load_tets == 1
        ids = str2num(tline);
        numTets = numTets + 1;
        if numTets > numTets_alloc
            Tet.tetFaces = [Tet.tetFaces, zeros(4,numTets_alloc)];
            numTets_alloc = numTets_alloc*2;
        end
        Tet.tetFaces(:, numTets) = ids(1:4)';
    end
end
fclose(f_id);
Tet.vertexPoss = Tet.vertexPoss(:,1:numPoss);
Tet.tetFaces = Tet.tetFaces(:,1:numTets);