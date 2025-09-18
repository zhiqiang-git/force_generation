function [matC] = MatElasticity(type)
matC = zeros(6,6);
if type == 1
    Y1 = 1352;
    Y2 = 822;
    Y3 = 822;
    G12 = 399;
    G23 = 370;
    G13 = 399;
    V12 = 0.3;
    V23 = 0.3;
    V13 = 0.3;
end
if type == 2
    Y1 = 16;
    Y2 = 6.3;
    Y3 = 6.3;
    G12 = 3.3;
    G23 = 3.6;
    G13 = 3.3;
    V12 = 0.3;
    V23 = 0.45;
    V13 = 0.3;
end
matC(1,1) = 1/Y1;
matC(2,2) = 1/Y2;
matC(3,3) = 1/Y3;
matC(4,4) = 1/G23;
matC(5,5) = 1/G13;
matC(6,6) = 1/G12;
matC(2,1) = -V12/Y1;
matC(1,2) = matC(2,1);
matC(3,1) = -V13/Y1;
matC(1,3) = matC(3,1);
matC(3,2) = -V23/Y2;
matC(2,3) = matC(3,2);

matC = inv(matC);