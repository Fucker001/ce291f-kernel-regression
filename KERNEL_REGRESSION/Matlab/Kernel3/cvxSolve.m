function [UStar, PhiStar, VStar] = cvxSolve(Outputs,Kernel,ShareOfTrainingSet)

global NumberOfPoints

%% Sets the training Set.
SizeOfTrainingSet = floor(ShareOfTrainingSet*NumberOfPoints);
A = Kernel(1:SizeOfTrainingSet,:);
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
PhiStar = VStar + norm(UStar,1);
disp('End of CVX.')
toc