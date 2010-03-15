% Solves the L1 regularized least square for a specific training set using
% CVX.
% Also calculates VStar.

function [UStar, PhiStar, VStar] = Solve(Kernel, SizeOfTrainingSet)

global Outputs

%% Sets the training Set.
A = Kernel(SizeOfTrainingSet,:);
b = Outputs(1,1:SizeOfTrainingSet)';
n = size(Kernel,2);

%% Optimizes using CVX.
cvx_begin;
    variable u(n);
    minimize( norm(b-A*u) + norm(u,1) );
cvx_end;

%% Sets the outputs.
UStar = u;
VStar = norm(b-A*u,2);
PhiStar = VStar + norm(u,1);