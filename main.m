% ML Project Steps
% Method 1 Pixel by Pixel estimator (compares each pixel to vocab of colors
% from images)
%   First convert training images to LAB, use data from Luminance
%       Create three classes from data
%           SVM to compare each test pixel to class
%               Colorize

% Method 2 Region Segmentation estimation
%   Segment training images
%       Create vocabulary from hues
%           %Svm regions of test images to vocab
%              Colorize

% Method One

% Create Color Space: Color based Segmentation Using K - Means Clustering
%

%% get features and prep images
clear all;
close all;

scale=0.2;
windowSize = 9;
numPermutations = 5000;
myfilter = fspecial('gaussian',[3 3],0.5);
nColors = 5;
features_all = [];
true_labels = [];
%imstring = 'calhouse_%04d.jpg';
imstring = 'image_%04d.jpg';
%imstring = 'simple_%04d.jpg';
train_index = 1;

rgbIm_train = im2double(imresize(imfilter(imread(sprintf(imstring,train_index)),myfilter),scale));
grayIm_train = rgb2gray(rgbIm_train);


[numRows_train, numCols_train] = size(grayIm_train);


labIm = rgb2lab(rgbIm_train);
ab = double(labIm(:,:,2:3)); % account for color information in 'a*b*' space
nrows = size(ab,1);
ncols = size(ab,2);
ab_re = reshape(ab,nrows*ncols,2);

% repeat the clustering to avoid local minima
[cluster_idx ,~] = kmeans(ab_re,nColors,'distance','sqEuclidean', ...
    'Replicates',10);

%true labels of original image as result of k-means
pixel_labels = reshape(cluster_idx,nrows,ncols);

% Discretize color space
colora = zeros(nColors,1);
colorb = zeros(nColors,1);
count = zeros(nColors,1);
for i = 1:numRows_train
    for j = 1:numCols_train
        n = pixel_labels(i,j);
        for k = 1:nColors
            if n == k
                colora(k) = colora(k) + labIm(i,j,2);
                colorb(k) = colorb(k) + labIm(i,j,3);
                count(k) = count(k) +1;
            end
        end
    end
end

% new color space
averagea = colora./ count;
averageb = colorb./ count;

%Randomly select training pixels from image and stores to array
perm = randsample(numel(grayIm_train),numPermutations);
[trainRow,trainCol] = ind2sub(size(grayIm_train),perm);

%get patches surrounding randomly selected pixels
windowCount = 1;
for index = 1:numPermutations
    %make sure pixel not along border
    if((trainRow(index) > (windowSize+1)/2)  && (trainRow(index) < numRows_train - (windowSize+1)/2) && (trainCol(index) > (windowSize+1)/2)  && (trainCol(index) < numCols_train - (windowSize+1)/2))
        %keep this pixel
        windowLabels(windowCount) = pixel_labels(trainRow(index),trainCol(index));
        windows{windowCount} = grayIm_train((trainRow(index)- (windowSize-1)/2):((trainRow(index) + (windowSize-1)/2)), ((trainCol(index)- (windowSize-1)/2):(trainCol(index) + (windowSize-1)/2)));
        windowCount = windowCount + 1;
    end
end

%calculate window features
for index = 1:(windowCount - 1)
    vdiff_w(index,1) = mean(mean(abs(diff(windows{index}))));
    hdiff_w(index,1) = mean(mean(abs(diff(windows{index}'))));
    L_w(index,1) = windows{index}((windowSize+1)/2,(windowSize+1)/2);
    var_w(index,1) = var(windows{index}(:));
    mean_w(index,1) = mean(windows{index}(:));
    median_w(index,1) = median(windows{index}(:));
    num_corners_w(index,1) = size(corner(windows{index}),1);
%     temp = detectMSERFeatures(windows{index});
%     num_MSER_feat_w(index,1) = temp.Count;
    sobelim = edge(windows{index},'Sobel',0.02);
    ratio_edges_w(index,1) = sum(sum(sobelim))/numel(sobelim);
end


features = [vdiff_w hdiff_w L_w var_w mean_w median_w num_corners_w ratio_edges_w]; % Redefine X as all features
true_labels = windowLabels';



%% train SVM with features
t = templateSVM('Standardize',1,'KernelFunction','gaussian');
classnames = cell(1,nColors);
for i = 1:nColors
    classnames{i} = num2str(i);
end
%pool = parpool; % Invoke workers
options = statset('UseParallel',1);
Mdl = fitcecoc(features,true_labels,'Coding','onevsall','Learners',t,'FitPosterior',1,...
    'ClassNames',classnames,'Verbose',2,'Options',options);


%% test model
test_index = 2;
rgbIm_test = im2double(imresize(imfilter(imread(sprintf(imstring,test_index)),myfilter),scale));
grayIm_test = rgb2gray(rgbIm_test);
[numRows_test, numCols_test] = size(grayIm_test);
est_image = zeros(numRows_test-windowSize+1,numCols_test-windowSize+1);
vdiff_w = est_image;
hdiff_w = est_image;
L_w = est_image;
var_w = est_image;
mean_w = est_image;
median_w = est_image;
num_corners_w = est_image;
ratio_edges_w = est_image;
for i = 1:numRows_test-windowSize+1
    for j = 1:numCols_test-windowSize+1
        im_patch = grayIm_test(i:i+windowSize-1,j:j+windowSize-1);
        %calculate window features
        vdiff_w(i,j) = mean(mean(abs(diff(im_patch))));
        hdiff_w(i,j) = mean(mean(abs(diff(im_patch'))));
        L_w(i,j) = im_patch((windowSize+1)/2,(windowSize+1)/2);
        var_w(i,j) = var(im_patch(:));
        mean_w(i,j) = mean(im_patch(:));
        median_w(i,j) = median(im_patch(:));
        num_corners_w(i,j) = size(corner(im_patch),1);
%         temp = detectMSERFeatures(im_patch);
%         num_MSER_feat_w(i,j) = temp.Count;
        sobelim = edge(im_patch,'Sobel',0.02);
        ratio_edges_w(i,j) = sum(sum(sobelim))/numel(sobelim);
    end
    disp(i)
end

test_features = [vdiff_w(:) hdiff_w(:) L_w(:) var_w(:) mean_w(:) median_w(:) num_corners_w(:) ratio_edges_w(:)]; % Redefine X as all features
%[label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);
[est_label,score] = predict(Mdl,test_features);
for i = 1:length(est_label)
    est_class(1,i) = str2num(est_label{i});
end
est_class_im = reshape(est_class,numRows_test-windowSize+1,numCols_test-windowSize+1);


%% check results
testIm = rgb2lab(rgbIm_test((windowSize+1)/2:end-(windowSize-1)/2,(windowSize+1)/2:end-(windowSize-1)/2,:));

for i = 1:size(est_class_im,1)
    for j = 1:size(est_class_im,2)
        n = est_class_im(i,j);
        for k = 1:nColors
            if n == k
                testIm(i,j,2) = averagea(k);
                testIm(i,j,3) = averageb(k);
            end
        end
    end
end

rgbTestIm = lab2rgb(testIm);
figure()
imshow(rgbTestIm)
%
