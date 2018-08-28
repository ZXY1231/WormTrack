function [imgx,imgy] = BgThresh(img,imgename)
% 
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version


% Choose the threshold of image
imgbwthresh = imbinarize(img,0.31);%previous use 0.2
imwrite(imgbwthresh, ['thresh/' imgename '.png']);

% Choose the reasonable area to find out worm
imgremovesmall = bwareaopen(imgbwthresh,70);   %remove regions <100 pixels
imgremoved = RemoveBigArea(imgremovesmall,600);%remove regions >600 pixels
%imwrite(imgremoved, imgename);
imwrite(imgremoved, ['ImgeProcessing/' imgename '.png']);


% Find connected components in binary image
CC = bwconncomp(imgremoved,26); %default is 8,18, 26 neighborhood also OK

% Due to cellfun limit, size of img must be a cell form
s = size(imgremoved);
SizeCell = cell(1,numel(CC.PixelIdxList));
SizeCell(1:end) = {s};

% Find out the centre of worm
centroids = cellfun(@LocateWormCentre, CC.PixelIdxList, SizeCell);
[x,y] = ind2sub(s, centroids); % Transfer linear index to subscript
imgx = y;
imgy = x;
imgy = s(2)-imgy; % What is mean? invert the image 

