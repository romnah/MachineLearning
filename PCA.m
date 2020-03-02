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
   imshow(J);
   
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

%Plotting pca1 and pca2
figure(2);
plot(pca1(1,:),pca2(1,:),'o');