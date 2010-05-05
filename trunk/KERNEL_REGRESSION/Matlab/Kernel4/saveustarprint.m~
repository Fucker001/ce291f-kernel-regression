%% save u star
for i=1:618
    if abs(UStar(i,1))<10^-4
        Ustarprint(i,1)=10^-4;
    else Ustarprint(i,1)=UStar(i,1);
    end
end

%% get phi
SizeOfTrainingSet = floor(0.5*309);
Uaround=UStar;
%Uaround(350,1)=-0.027107842485749;
Uaround=Ustarprint;
phi=getPhi(Uaround,Kernel(1:SizeOfTrainingSet,:),Outputs(1,1:SizeOfTrainingSet)');
disp(phi);

%% Plot along one u
SizeOfTrainingSet = floor(0.5*309);
%k=352;
k=521;
Uaround=UStar;
precision_size=1000;
x=zeros(precision_size);
Phis=zeros(precision_size);
for i=1:precision_size
    x(i)=(i-500)/10;
    Uaround(k,1)=x(i);
    Phis(i)=getPhi(Uaround,Kernel(1:SizeOfTrainingSet,:),Outputs(1,1:SizeOfTrainingSet)');
end
figure
plot(x,Phis);
axis([-5 5 780 786]);

%% plot fat k row major
row=170;
column=326;

figure
plot(fatKrowmajor(row,:),'r');
figure
plot(Kernel(row,:),'b');

figure
plot(fatKrowmajor(:,column),'r');
figure
plot(Kernel(:,column),'b');

%% plot fat k column major

fatKColumnMajorOneRow=zeros(1,618*309);
KernelOneRow=zeros(1,618*309);
PlotDiff=zeros(1,618*309);
for i=1:size(fatKcolmajor,1)
    for j=1:size(fatKcolmajor,2)
        fatKColumnMajorOneRow(1,i+j*309)=fatKcolmajor(i,j);
        KernelOneRow(1,i+j*309)=Kernel(i,j);
        PlotDiff(1,i+j*309)=abs(fatKColumnMajorOneRow(1,i+j*309)-KernelOneRow(1,i+j*309));
    end
end
%% plot
figure
plot(PlotDiff,'r');


%% max difference
maxdiff=0;I=0;J=0;
for i=1:size(fatKrowmajor,1)
    for j=1:size(fatKrowmajor,2)
        diff=abs(fatKrowmajor(i,j)-Kernel(i,j));
        if diff>maxdiff && j~=324
            maxdiff=diff;
            I=i;
            J=j;
        end
    end
end
I
J
maxdiff

%% read files for the gaussian
gaussianInitial=csvread('gaussian_kernel.txt');
gaussianNormalized=csvread('gaussian_kernel_trace_normalized.txt');
gaussianSVDed=csvread('gaussian_kernel_SVDed.txt');
gaussianS=csvread('gaussian_kernel_S.txt');
gaussianU=csvread('gaussian_kernel_U.txt');
gaussianSsquareroot=csvread('gaussian_kernel_S_squareroot.txt');
KernelStar=csvread('KernelStar.txt');

%% gaussian initial plot
fatKColumnMajorOneRow=zeros(1,309*309);
KernelOneRow=zeros(1,309*309);
PlotDiff=zeros(1,309*309);
for i=1:309
    for j=1:309
        fatKColumnMajorOneRow(1,i+j*309)=prenormalization(i,j);
        KernelOneRow(1,i+j*309)=gaussianInitial(i,j);
        PlotDiff(1,i+j*309)=abs(fatKColumnMajorOneRow(1,i+j*309)-KernelOneRow(1,i+j*309));
    end
end
figure
plot(PlotDiff,'r');
%% gaussian normalized plot
fatKColumnMajorOneRow=zeros(1,309*309);
KernelOneRow=zeros(1,309*309);
PlotDiff=zeros(1,309*309);
for i=1:309
    for j=1:309
        fatKColumnMajorOneRow(1,i+j*309)=presvd(i,j);
        KernelOneRow(1,i+j*309)=gaussianNormalized(i,j);
        PlotDiff(1,i+j*309)=abs(fatKColumnMajorOneRow(1,i+j*309)-KernelOneRow(1,i+j*309));
    end
end
figure
plot(PlotDiff,'r');

%% gaussian svded plot

fatKColumnMajorOneRow=zeros(1,309*309);
KernelOneRow=zeros(1,309*309);
PlotDiff=zeros(1,309*309);
maxdiff=0;
for i=1:309
    for j=1:309
        fatKColumnMajorOneRow(1,i+j*309)=final(i,j);
        KernelOneRow(1,i+j*309)=gaussianSVDed(i,j);
        PlotDiff(1,i+j*309)=abs(fatKColumnMajorOneRow(1,i+j*309)-KernelOneRow(1,i+j*309));
%         diff=abs(final(i,j)-gaussianSVDed(i,j));
%         if diff>maxdiff
%             maxdiff=diff;
%         end
    end
end
figure
plot(PlotDiff,'r');
% disp(maxdiff);

%% gaussian svd U plot

fatKColumnMajorOneRow=zeros(1,309*309);
KernelOneRow=zeros(1,309*309);
PlotDiff=zeros(1,309*309);
for i=1:309
    for j=1:309
        fatKColumnMajorOneRow(1,i+j*309)=U(i,j);
        KernelOneRow(1,i+j*309)=gaussianU(i,j);
        PlotDiff(1,i+j*309)=abs(fatKColumnMajorOneRow(1,i+j*309)-KernelOneRow(1,i+j*309));
    end
end
figure
plot(PlotDiff,'r');

%% gaussian S
dif=zeros(309);
for i=1:309
    dif(i)=abs(s(i)-gaussianS(i,i));
end
figure
plot(dif,'r')

%% plot estimates
figure
%Red: Training set
%scatter3(Inputs(1,1:train),Inputs(3,1:train),Outputs(1:train),'r', 'filled');
%hold on
%Green: Test set, Actual values
%scatter3(Inputs(1,train+1:total),Inputs(3,train+1:total),Outputs(1,train+1:total),'g', 'filled');
%hold on
%Blue: Test set, Estimated values
scatter3(Inputs(1,155:309),Inputs(3,155:309),estimates,'b', 'filled');
axis([0 1 0 1 0 1000]);
xlabel('time in the day');
ylabel('Weather status');
zlabel('travel time (s)')
grid on
axis square

%% 


