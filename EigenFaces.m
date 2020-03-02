clc;
A = zeros(154,1600);

Files=dir('C:\Users\ROMIT\Desktop\Drexel Study\Quarter 2\CS 613 Machine Learning\Week 1\Assignment 1\yalefaces\subject*');
for k=1:length(Files)
   FileNames=Files(k).name;
   F = imread(strcat('C:\Users\ROMIT\Desktop\Drexel Study\Quarter 2\CS 613 Machine Learning\Week 1\Assignment 1\yalefaces\', FileNames));
   %fprintf(1,'%s \n',strcat('C:\Users\ROMIT\Desktop\Drexel Study\Quarter 2\CS 613 Machine Learning\Week 1\Assignment 1\yalefaces\', FileNames));
   %imshow(F);
   
   %resizing the images to 40x40
   J = imresize(F,[40 40]); %resizing F to 40x40
   %imshow(J);
   
   %Flattening the 40x40 image into a signle 1x1600 array
   FlattenJ = reshape(J, 1, 1600); %converting 40x40 to 1x1600

   %Concatenating all the 1x1600 images into single matrix of size 154x1600
   A(k,:)= FlattenJ;
end

%Calculating Mean and Standard Deviation of the vector A
m = mean(A); %get mean of each feature
s = std(A); %get std of each feature
%m = 3.2500 237.5000
%s = 1.2583 368.2730

%Standarising the Data
A = A - repmat(m,size(A,1),1);
A = A./repmat(s,size(A,1),1);

%Calculating Covariance of the matrix
covA = cov(A);

%Calculating Eigen vector and Eigen value
[vec, val] = eig(covA);

%Finding Max Eigen Value, its position (column), taking that column out of
%Eigen vector and multiplying it to data matrix to evaluate pca1
max1val = max(val,[],'all'); %find max position. take column and pick the same column from vec and multiply it to A.
[rowMax1, colMax1] = find(val==max1val);
vecColMax1 = vec(:, colMax1);
pca1 = A*vecColMax1;
pca1 = pca1';

%Finding second Max Eigen Value, its position (column), taking that column out of
%Eigen vector and multiplying it to data matrix to evaluate pca2
max2val = max(val(val<max(val, [], 'all')), [], 'all');
[rowMax2, colMax2] = find(val==max2val);
vecColMax2 = vec(:, colMax2);
pca2 = A*vecColMax2;
pca2 = pca2';

%sort the eigenvalues and eigenvectors
[d,index] = sort(diag(val));
eigValSorted = val(index,index);
eigVecSorted = vec(:,index);

%find the principal component
vecProject = eigVecSorted(:,1600);
E = A(1,:)*vecProject; %using principal component

%no. of principal components required
percent = 0;
eigValSum = 0;
npc = 0;
sum = trace(val);
k =1600;
while (percent<=0.95)
    eigValSum = eigValSum + val(k,k);
    percent = eigValSum/sum;
    npc = npc+1;
    k = k-1;
end

%main image
 mainImg = imread(strcat('C:\Users\ROMIT\Desktop\Drexel Study\Quarter 2\CS 613 Machine Learning\Week 1\Assignment 1\yalefaces\subject02.centerlight'));
 mainJ = imresize(mainImg,[40 40]);
 figure(10);
 imshow(mainJ);
 
%visualise most important component as image
sampleImage1 = reshape(vecProject,[40,40]);
figure(5);
imshow(sampleImage1,[min(vecProject),max(vecProject)]);

% Reconstruction using principal vector
ReC_pVec = E*(vecProject.');
img2 = reshape(ReC_pVec,[40,40]);
figure(4);
imshow(img2,[min(ReC_pVec),max(ReC_pVec)]);

%Reconstruction using npc principal components 
NPC = 1600-npc+1;
vecProj2 = eigVecSorted(:,NPC:1600);
E1 = A(1,:)*vecProj2;%using npc components
ReC_pVec1 = E1*(vecProj2');
sampleImage3 = reshape(ReC_pVec1,[40,40]);
figure(3);
imshow(sampleImage3,[min(ReC_pVec1),max(ReC_pVec1)]);
