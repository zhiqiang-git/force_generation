function [stress_ms] = StressProcessing(OPT, displaceField, type)
% Compute the max norm of the vertex based stree field
stress_tensor = OPT.D*OPT.B*displaceField;
numT = length(stress_tensor)/6;
stress_ms = zeros(1, numT);
for tId = 1 : numT
    vec = stress_tensor((6*tId-5):(6*tId));
    sigma = [vec(1), vec(6), vec(5);
        vec(6), vec(2), vec(4);
        vec(5), vec(4), vec(3)];
    if type ~= 2
        stress_ms(tId) = max(svd(full(sigma)));
    else
        sign = sum(vec(1:3)) > 0;
        stress_ms(tId) = (2*sign-1)*sqrt(sum(sum(sigma.*sigma)));
    end
end