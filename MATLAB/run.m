% Before running MATLAB code, please cvx_setup
fileDir = mfilename('fullpath');
here = fileparts(fileDir);
utilsDir = fullfile(here, 'utils');
scriptsDir = fullfile(here, 'scripts');
addpath(genpath(utilsDir));
addpath(genpath(scriptsDir));

root = '~/Documents/Physical-stability/main/python/data/chair_1200';
pattern = '**/tet/tetmesh.m';
files = dir(fullfile(root, pattern));

maxTime = 600; % seconds = 10 minutes

for i = 1:numel(files)
    name = fullfile(files(i).folder, files(i).name);
    save_name = fullfile(files(i).folder, 'WCSA.mat');

    if ~isfile(name)
        fprintf('File not found: %s\n', name);
        continue;
    elseif isfile(save_name)
        fprintf('Skip existing file: %s\n', save_name);
        continue; 
    else
        fprintf('Finding the file: %s\n', name);
    end

    % Run processing in parallel worker with timeout
    f = parfeval(@processOneShape, 1, name, save_name);

    try
     % First, wait up to maxTime
          wait(f, 'finished', maxTime);

          if strcmp(f.State,'finished')
               result = fetchOutputs(f);  % now safe to collect
               fprintf('Process: %s finished with status %d\n', name, result);
          else
               fprintf('Timeout exceeded for %s — skipping.\n', name);
               cancel(f);
          end
     catch ME
          fprintf('Error processing %s: %s\n', name, ME.message);
          cancel(f);
     end
end

%% === Helper function ===
function status = processOneShape(name, save_name)
    Tet = LoadTetMesh(name);
    Tet = TetDataCompute(Tet);
    type = 1;
    numEigens = 15;
    [OPT, Basis] = EigenModeAnalysis(Tet, type, numEigens);
    eps = 0.025;
    M_r = 5;
    OPT = WCSA_Prepare(Tet, Basis, OPT, eps, M_r);
    p_max = 1e2;
    F_tot_ratio = 1;

    [score_opt, p_opt, stress_opt, F_tot_opt, eigenmode_id, weakregion_id, ...
        u_all, p_all, stress_all] = WCSA(OPT, p_max, F_tot_ratio, 0, 0);

    save(save_name, 'Basis', 'OPT', 'eigenmode_id', 'weakregion_id', ...
        'p_all', 'u_all', 'stress_all', 'Tet');
     status = 1;
end
