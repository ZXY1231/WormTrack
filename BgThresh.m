function [imgx, imgy,img_thresh, img_removed] = BgThresh(img_soure, thresh, small, big)
% Extract worm location from normalized image
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version
% | 0.2     | H.F.   | 18.09.05 |
% To Do: Binarize image with locally adaptive thresholding or only take
% threshold but keep graydrade
%

% Choose the threshold of image
img_thresh = imbinarize(img_soure,thresh);%previous use 0.2

% Choose the reasonable area to find out worm
imgremovesmall = bwareaopen(img_thresh,small);   %remove regions <100 pixels
img_removed = RemoveBigArea(imgremovesmall,big);%remove regions >600 pixels

% Find connected components in binary image
CC = bwconncomp(img_removed,26); %default is 8,18, 26 neighborhood also OK

% Due to cellfun limit, size of img must be a cell form
s = size(img_removed);
SizeCell = cell(1,numel(CC.PixelIdxList));
SizeCell(1:end) = {s};

% Find out the centre of worm
centroids = cellfun(@LocateWormCentre, CC.PixelIdxList, SizeCell);
[x,y] = ind2sub(s, centroids); % Transfer linear index to subscript
imgx = y;
imgy = x;
imgy = s(2)-imgy; % What is mean? invert the image 

