
%% (INIT 1) Get the actual dataset, shape it, and plot it, write it down in csv.

% input.txt and output.txt have the input actual dataset.
in = csvread('input.txt');
out= csvread('output.txt');

x1=in(1,:);
x2=in(2,:);
x3=in(3,:);
z_def=out(1,:);
z=out(1,:);

si=size(x1,2);
for i=1:si
    z(1,i)=90+(1+5*normpdf(x1(1,i),0.3,0.04)+10*normpdf(x1(1,i),0.7,0.10)+200*x3(1,i))*z(1,i)/60;
end

scatter3(x1,x3,z,'filled');
axis([0 1 0 1 0 1000]);
xlabel('time in the day');
ylabel('Weather status');
zlabel('travel time (s)')
grid on
axis square

csvwrite('good_dataset_input.txt',in);
csvwrite('good_dataset_output.txt', z);

%% (INIT 2) filter to get a sub-dataset (smaller), and plot it, save it.
x1f=[];
x2f=[];
x3f=[];
zf=[];
max=size(x1,2);

counter=0;
for i=1:max
    counter = counter + 1;
    if counter>9
       x1f=[x1f x1(i)];
       x2f=[x2f x2(i)];
       x3f=[x3f x3(i)];
       zf=[zf z(i)];
       counter=0;
    end
end

% Save it by reorganizing it first...
inf=[x1f;x2f;x3f];
sizeinf=size(inf,2);
for i=1:sizeinf
    new_index = floor(rand*sizeinf);
    if new_index==0
        new_index=1;
    end
    % input
    temp=inf(:,i);
    inf(:,i)=inf(:,new_index);
    inf(:,new_index)=temp;
    % output
    temp2=zf(:,i);
    zf(:,i)=zf(:,new_index);
    zf(:,new_index)=temp2;
end
csvwrite('inputs.txt',inf);
csvwrite('outputs.txt', zf);

% plot
figure
scatter3(inf(1,:),inf(3,:),zf,'filled');
axis([0 1 0 1 0 1000]);
xlabel('time in the day');
ylabel('Weather status');
zlabel('travel time (s)')
grid on
axis square

%% Kernel Regression, and plot of the results
[Error,AggregatedError,Estimate,UStar,Inputs,Outputs,ShareOfTrainingSet,Kernel] = kernelRegression(0.5,0.01);
total=size(Inputs,2);
est_size=size(Estimate,2);
train=floor(total*ShareOfTrainingSet);
disp 'Error on all points:';disp(AggregatedError(1));
disp 'Error on training:';disp(AggregatedError(2));
disp 'Error on test:';disp(AggregatedError(3));

csvwrite('estimate_on_all.txt',Estimate);
csvwrite('aggregated_error.txt',AggregatedError);

figure
%Red: Training set
scatter3(Inputs(1,1:train),Inputs(3,1:train),Outputs(1:train),'r', 'filled');
hold on
%Green: Test set, Actual values
scatter3(Inputs(1,train+1:total),Inputs(3,train+1:total),Outputs(1,train+1:total),'g', 'filled');
hold on
%Blue: Test set, Estimated values
scatter3(Inputs(1,total-est_size+1:total),Inputs(3,total-est_size+1:total),Estimate,'b', 'filled');
axis([0 1 0 1 0 1000]);
xlabel('time in the day');
ylabel('Weather status');
zlabel('travel time (s)')
grid on
axis square

figure
usize=size(UStar(:,1));
plot(1:usize,UStar);
xlabel('Kernel Eigenvector');
ylabel('Optimal amplitude (UStar)');
hold on
steps=zeros(usize);
small=size(Inputs(1,:),2);
for i=1:usize
   steps(i)=3*floor(i/small);
end
plot(1:usize,steps);

%% replot

figure
%Red: Training set
scatter3(inf(1,1:train),inf(3,1:train),zf(1:train),'r', 'filled');
hold on
%Green: Test set, Actual values
% scatter3(inf(1,train+1:total),inf(3,train+1:total),zf(1,train+1:total),'g', 'filled');
% hold on
%Blue: Test set, Estimated values
scatter3(inf(1,train+1:total),inf(3,train+1:total),Estimate,'b', 'filled');
axis([0 1 0 1 0 1000]);
xlabel('time in the day');
ylabel('Weather status');
zlabel('travel time (s)')
grid on
axis square

%% Calculate aggregated error for training and test set depending on share of training set.
iterations=30;
results=zeros(5,1);
for i=1:iterations
    [Error,AggregatedError,Estimate,UStar,Inputs,Outputs,ShareOfTrainingSet]...
        = kernelRegression(i/iterations,0.01);
    results=[results AggregatedError];
end
results=results(:,2:size(results,2));

%% Plot aggregated error on share of training set
figure
plot(results(1,:),results(3,:),'r');
hold on
plot(results(1,:),results(4,:),'g');
xlabel('Share of training data');
ylabel('Total error (%)');
h = legend('Error on the training set','Error on the test set',2);
set(h,'Interpreter','none')

%% Calculate aggregated error for training and test set depending on rho.
span=0.7;
divide=100;
results=zeros(5,1);
for i=0.01:span/divide:span
    [Error,AggregatedError,Estimate,UStar,Inputs,Outputs,ShareOfTrainingSet]...
        = kernelRegression(0.2,i);
    results=[results AggregatedError];
end
results=results(:,2:size(results,2));

%% Plot aggregated error on rho
figure
plot(results(5,:),results(3,:),'r');
hold on
plot(results(5,:),results(4,:),'g');
xlabel('Weight of normalization against over-fitting');
ylabel('Total error (%)');
h = legend('Error on the training set','Error on the test set',2);
set(h,'Interpreter','none')


%% filter
xf=[];
yf=[];
zf=[];
max=size(x1,2);
window = 1;
min_w = 0;

counter=0;
for i=1:max
    counter = counter+1;
    if (y(i) >= min_w && y(i) <= min_w+window && counter>0)
        xf=[xf x(i)];
        yf=[yf y(i)];
        zf=[zf z(i)];
        counter=0;
    end
end

%% scatter 2d plot

figure
scatter(xf,zf,'filled')
axis([0 1 0 1000])
xlabel('time in the day');
ylabel('travel time (s)');
grid on
axis square

%% tt
figure
scatter(tt(:,1),tt(:,2),'filled','r');


