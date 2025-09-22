% Before running MATLAB code, please cvx_setup
fileDir = mfilename('fullpath');
here = fileparts(fileDir);
utilsDir = fullfile(here, 'utils');
scriptsDir = fullfile(here, 'scripts');
addpath(genpath(utilsDir));
addpath(genpath(scriptsDir));

root = '~/Documents/Physical-stability/main/python/data/chair_1200';
pattern = '**/sdf/tetmesh.m';
files = dir(fullfile(root, pattern))

for i = 1:numel(files)
    name = fullfile(files(i).folder, files(i).name);
    save_name = fullfile(files(i).folder, 'WCSA.mat');
    %name = sprintf('data/test.m');
    %save_name = sprintf('data/result.mat');
    Tet = LoadTetMesh(name);
    %Tet = loadTetMesh('Data/Deformed/deformed_0.m');
    Tet = TetDataCompute(Tet);

    type = 1;
    numEigens=15;
    [OPT, Basis] = EigenModeAnalysis(Tet, type, numEigens);
    eps = 0.025;
    M_r = 5;
    OPT = WCSA_Prepare(Tet, Basis, OPT, eps, M_r);
    p_max = 1e2;
    F_tot_ratio = 1;
    [score_opt, p_opt, stress_opt, F_tot_opt, eigenmode_id, weakregion_id, u_all, p_all, stress_all]...
         = WCSA(OPT, p_max, F_tot_ratio, 0, 0);
    save(save_name, 'Basis', 'OPT', 'eigenmode_id', 'weakregion_id', 'p_all', 'u_all', 'stress_all', 'Tet');
    fprintf('Process: %s finished\n', name);
