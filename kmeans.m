clc


% Specify the folder where the files live.
myFolder = 'C:\Users\ROMIT\Desktop\Drexel Study\Quarter 2\CS 613 Machine Learning\Week 1\Assignment 1\yalefaces';

imArray = zeros(154,1600,'uint8');
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, 'subject*'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName)
  % reading filename in as an image array with imread()
  imageArray = imread(fullFileName);
  resizedimage = imresize(imageArray, [40 40]);
  reshapedImage = reshape(resizedimage,[1,1600]);
  
  imArray(k,1:1600) = reshapedImage;
  %imshow(imageArray);  % Display image.
  %drawnow; % Force display to update immediately.
end

%Find Mean of the matrix
M = mean(double(imArray));
%Find Std deviation of the matrix
S = std(double(imArray));
%Standardise the matrix
imArray = double(imArray) - repmat(M,size(imArray,1),1);
imArray = double(imArray)./repmat(S,size(imArray,1),1);

kmyClusters = 2;
myKMeans(imArray,kmyClusters);
%myKMeans function with input dataset X and no. of myClusters nocl
function kmeans = myKMeans(X,kmyClusters)

%m=number of observations, n = no.of features
[row,col]= size(X);

  if col > 3
    %Find covariance of the standardised matrix
        C = cov(X);
        [vec,val] = eig(C);
        diagVal = diag(val);
        % sort eigenvalues in descending order
        diagVal = diagVal(end:-1:1);
        
        %
        B = []; 
        [MaxVal,In] = maxk(val(:),3);
        for c = 1:3
            [I_rowk, I_colk] = ind2sub(size(val),In(c));
            W = vec(:,I_colk);

            for i=1:size(W,3)
              B = [B, W(:,:,i)];
            end
        end
    end
    
    Z = X * B ;
  
  %generate random indices
  rng(0);
  rndmIdx = randi([1,row],1,kmyClusters);
  
  %reference vectors
  refV = Z(rndmIdx,:);
  
  %Eucledian Distance
  for k = 1:kmyClusters
      for i = 1:row
          diff = Z(i,:)-refV(k,:);
          dist(k,i) = norm(diff); 
      end
  end
    
  %divide into myClusters
  [min_val,index] = min(dist);
  myCluster1 =[];
  myCluster2 =[];
  for i = 1:row
      if index(1,i) == 1
          myCluster1 = [myCluster1;Z(i,:)];
      else 
          myCluster2 = [myCluster2;Z(i,:)];
      end  
  end
  
  %new reference vector
  meanC1 = mean(myCluster1);
  meanC2 = mean(myCluster2);
  newRefV = [meanC1;meanC2];

  %manhattan distance
  mndst = sum(abs(newRefV-refV).');
  mndst = mndst.';
  
  e_dist = sum(mndst);
  e = 2^(-23);
  
  iterations = 1;
  figure
  plot3(myCluster1(:,1),myCluster1(:,2),myCluster1(:,3), 'x', 'Color', 'red');
  F(1)=getframe(gcf);
  hold on
  plot3(myCluster2(:,1),myCluster2(:,2),myCluster2(:,3), 'x', 'Color', 'blue');
  F(2)=getframe(gcf);
  plot3(meanC1(:,1),meanC1(:,2),meanC1(:,3), 'o', 'MarkerFaceColor', 'red');
  F(3)=getframe(gcf);
  plot3(meanC2(:,1),meanC2(:,2),meanC2(:,3), 'o', 'MarkerFaceColor', 'blue');
  F(4)=getframe(gcf);
  title(iterations)
  hold off
  
  refV = newRefV;
    
  %iterate until we get final results
  while (e_dist>e)
      for k = 1:kmyClusters
          for i = 1:row
          diff = Z(i,:)-refV(k,:);
          dist(k,i) = norm(diff); 
          end
      end
      %divide into myClusters
      [min_val,index] = min(dist);
      myCluster1 =[];
      myCluster2 =[];
      for i = 1:row
          if index(1,i) == 1
              myCluster1 = [myCluster1;Z(i,:)];
          else
              myCluster2 = [myCluster2;Z(i,:)];
          end
      end
      
      newRefV = [];
      meanC1 = mean(myCluster1);
      meanC2 = mean(myCluster2);
      newRefV = [meanC1;meanC2];
      
      mndst = sum(abs(newRefV-refV).');
      mndst = mndst.';
      e_dist = sum(mndst);
      refV = newRefV;
      Z = [myCluster1;myCluster2];
      iterations = iterations+1;
  
      figure
      plot3(myCluster1(:,1),myCluster1(:,2),myCluster1(:,3), 'x', 'Color', 'red');
      F(1)=getframe(gcf);
      hold on
      plot3(myCluster2(:,1),myCluster2(:,2),myCluster2(:,3), 'x', 'Color', 'blue');
      F(2)=getframe(gcf);
      plot3(meanC1(:,1),meanC1(:,2),meanC1(:,3), 'o', 'MarkerFaceColor', 'red');
      F(3)=getframe(gcf);
      plot3(meanC2(:,1),meanC2(:,2),meanC2(:,3), 'o', 'MarkerFaceColor', 'blue');
      F(4)=getframe(gcf);
      title(iterations)
      hold off
      
  end
  
video=VideoWriter('kmeans.avi');
video.FrameRate = 0.3;
open(video);
writeVideo(video,F);
close(video);
end