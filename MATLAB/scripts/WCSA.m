function [score_opt, p_opt, stress_opt, F_tot_opt, eigenmode_id, weakregion_id, u_all, p_all, stress_all]...
    = WCSA(OPT, p_max, F_tot_ratio, eigen_id, weak_id)
%
numT = size(OPT.D, 2)/6;
numV = size(OPT.B, 2)/3;
numBV = size(OPT.mat_SigmaNA, 2);
numEigenModes = size(OPT.tetWeights, 1);
numWeakRegions = size(OPT.weakRegions{1}, 2);
%
C_star = sparse(3*numV+6, 3*numV+6);
C_star(1:(3*numV),1:(3*numV)) = OPT.L;
C_star((3*numV+1):(3*numV+6),1:(3*numV)) = [OPT.mat_Sigma; OPT.mat_SigmaT];
C_star(1:(3*numV), (3*numV+1):(3*numV+6)) = [OPT.mat_Sigma; OPT.mat_SigmaT]';
% mat_tp = [(kron(OPT.tetWeights, ones(1,6))*OPT.V*OPT.D*OPT.B)';zeros(6,numEigenModes)];

% 4.
% Zhiqiang: tetWeights serves both as weights and trace operator
trace_operator = [ones(1, 3), zeros(1, 3)];
rows = zeros(1, 0);
cols = zeros(1, 0);
vals = zeros(1, 0);
for eigenId = 1 : numEigenModes
    for regionId = 1 : numWeakRegions
        numTIds = size(OPT.weakRegionTets{eigenId}{regionId}, 2);
        rows = [rows, ...
            ((eigenId-1)*numWeakRegions+regionId)*ones(1, numTIds)];
        cols = [cols, ...
            OPT.weakRegionTets{eigenId}{regionId}];
        vals = [vals, ...
            OPT.tetWeights(eigenId, OPT.weakRegionTets{eigenId}{regionId})];

        % weights((eigenId-1)*numWeakRegions+regionId, OPT.weakRegions{eigenId}{regionId}) = ...
        %     OPT.tetWeights(EigenId, OPT.weakRegions{eigenId}{regionId});
    end
end
weights = sparse(rows, cols, vals, numEigenModes*numWeakRegions, numT);

mat_tp = [(kron(weights, trace_operator)*OPT.V*OPT.D*OPT.B)';...
    zeros(6,numEigenModes*numWeakRegions)];
mat_q = C_star\mat_tp;
mat_f = OPT.mat_NA'*mat_q(1:(3*numV),:);

% Compute the maximum area of all weeakregions
% maxWeakRegionArea = 0;
% for eigenId = 1 : numEigenModes
%     for i = 1 : numWeakRegions
%         activeBVIds = OPT.weakRegions{eigenId}{i};
%         A = OPT.A(activeBVIds, activeBVIds);
%         sumArea = full(sum(diag(A)));
%         if sumArea > maxWeakRegionArea
%             maxWeakRegionArea = sumArea;
%         end
%     end
% end
% F_tot_max = F_tot_ratio*maxWeakRegionArea;
F_tot_max = F_tot_ratio*full(sum(diag(OPT.A)));

% define some variables to save all optimization results
u_all = zeros(0, 3*numV);
p_all = zeros(0, numBV);
stress_all = zeros(0, numT);

score_opt = 0;
F_tot_opt = 0;
eigenmode_id = 0;
weakregion_id = 0;

if eigen_id == 0
    for eigenId = 1 : numEigenModes
        for i = 1 : numWeakRegions
            % activeBVIds = OPT.weakRegions{eigenId}{i};
            % A = OPT.A(activeBVIds, activeBVIds);
            % sumArea = sum(diag(A));
            % F_tot = min(F_tot_max, sumArea*p_max/1.5);
            % numVars = length(activeBVIds);
    
            % 1.
            % Zhiqiang: That's the problem
            % The Optimized force is not limited to the weak region
            % Global force can cause stress in the weak region
            % activeVIds = OPT.weakRegions{eigenId}{i};
            sumArea = sum(diag(OPT.A));
            F_tot = min(F_tot_max, sumArea*p_max/1.5);
            numVars = length(diag(OPT.A));
    
            % 2.
            % Zhiqiang: The weak region extraction has problem
            % weak region only have one index, weird
    
            % Zhiqiang: Only normal pressure is used, so p is a scalar
            cvx_begin quiet
                % variable p(numVars);
                % maximize sum(mat_f(activeBVIds,eigenId).*p)
                % subject to
                % 0 <= p;
                % p <= p_max;
                % OPT.mat_SigmaNA(:, activeBVIds)*p == 0;
                % OPT.mat_SigmaTNA(:, activeBVIds)*p == 0;
                % sum(A*p) == F_tot;
    
                variable p(numVars);
                % Zhiqiang: This is the article's error
                % Ku = f_ext = -NAp, not Ku = NAp
                maximize -sum(mat_f(:, (eigenId-1)*numWeakRegions+i).*p)
                subject to
                0 <= p;
                p <= p_max;
                OPT.mat_SigmaNA*p == 0;
                OPT.mat_SigmaTNA*p == 0;
                sum(OPT.A*p) == F_tot;
            cvx_end
            if strcmp(cvx_status, 'Infeasible') == 1
                continue;
            end
            % TetVF = C_star\[OPT.mat_NA(:, activeBVIds)*p;zeros(6,1)];
            % TetVF = TetVF(1:(3*numV));
            % stress_ms = stressMaxSingular_FE(OPT, TetVF, 1);
            % score = max(stress_ms)/F_tot;
            % if score > score_opt
            %     score_opt = score;
            %     p_opt = zeros(1, numBV);
            %     p_opt(activeBVIds) = p;
            %     F_tot_opt = F_tot;
            % end
    
            u_opt = C_star\[OPT.mat_NA*p;zeros(6,1)];
            u_opt = u_opt(1:(3*numV));
            stress_ms = StressProcessing(OPT, u_opt, 1);
            score = max(stress_ms);
            if score > score_opt
                score_opt = score;
                p_opt = p;
                stress_opt = stress_ms;
                F_tot_opt = F_tot;
                eigenmode_id = eigenId;
                weakregion_id = i;
            end
    
            u_all = [u_all; u_opt'];
            p_all = [p_all; p'];
            stress_all = [stress_all; stress_ms];
    
        end
    end
% Using perpendicular constrains to get more sets of external forces
else 
    eigenId = eigen_id;
    i = weak_id;
    sumArea = sum(diag(OPT.A));
    F_tot = min(F_tot_max, sumArea*p_max/1.5);
    numVars = length(diag(OPT.A));
    for j = 1 : 5
        cvx_begin quiet
            variable p(numVars);
            maximize -sum(mat_f(:, (eigenId-1)*numWeakRegions+i).*p)
            subject to
            0 <= p;
            p <= p_max;
            OPT.mat_SigmaNA*p == 0;
            OPT.mat_SigmaTNA*p == 0;
            sum(OPT.A*p) == F_tot;
            p_all*p == 0;
        cvx_end
        if strcmp(cvx_status, 'Infeasible') == 1
            disp(cvx_status);
            continue;
        end
    
        u_opt = C_star\[OPT.mat_NA*p;zeros(6,1)];
        u_opt = u_opt(1:(3*numV));
        stress_ms = StressProcessing(OPT, u_opt, 1);
        score = max(stress_ms);
        if score > score_opt
            score_opt = score;
            p_opt = p;
            stress_opt = stress_ms;
            F_tot_opt = F_tot;
            eigenmode_id = eigenId;
            weakregion_id = i;
        end
    
        u_all = [u_all; u_opt'];
        p_all = [p_all; p'];
        stress_all = [stress_all; stress_ms];
    end
end
