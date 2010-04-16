%Control for the regression

%% Initialization of Matlab
clear
clc
%% Test data
n1=100;
% O=[];
% I=[];
O=zeros(1,n1);
I=zeros(2,n1);
for i=1:n1
    I(1,i)=i;
    I(2,i)=2*i;
    I(3,i)=5*i;
    I(4,i)=-i;
    O(i)=3.5*i+6;
end
csvwrite('inputs.txt',I)
csvwrite('outputs.txt',O)
%% Regression
[Error,Estimate,Inputs,Outputs,ShareOfTrainingSet] = kernelRegression;
%% Total error
TotalError = mean(Error);
disp('This is your total error:')
disp(TotalError);
%% Plots
NumberOfPoints = size(Inputs,2);
SizeOfTrainingSet = floor(ShareOfTrainingSet*NumberOfPoints);
plot(Inputs(1,SizeOfTrainingSet+1:NumberOfPoints),...
    Outputs(1,SizeOfTrainingSet+1:NumberOfPoints),...
    Inputs(1,SizeOfTrainingSet+1:NumberOfPoints),Estimate);
legend('Reality','Estimate');