function [imgx,imgy] = BgThresh(img,imgename)
%img = imread('/Users/apple/Desktop/IGEM/post/tracking/WormTrack/contrastAdjusted.tif');
%img = img(600:end-600,600:end-600);
%max(img)
%min(img)
%imgbw = imbinarize(img,0.2);
%imgbw = rangefilt(img);
imgbwthresh = imbinarize(img,0.2);
%imgbw = multithresh(img,2);
%max(imgbw)
%min(imgbw)
size(img);
%figure(1)
%imshow(imgbw)
%figure(2)
%imshow(imgbwthresh)
%surf(img)

imgremovesmall = bwareaopen(imgbwthresh,200);%remove regions with less than 100 pixels
imgremoved = RemoveBigArea(imgremovesmall,600);%remove regions with more than 600 pixels
CC = bwconncomp(imgremoved,26);%default is 8,18, 26  are also OK
%centroids = CC.PixelIdxList;
%centroids = cellfun(@numel,CC.PixelIdxList)
%max(centroids)
%min(centroids)
%[biggest,idx] = max(centroids)
numel(CC.PixelIdxList);
%CC.PixelIdxList{idx};%cellµÄÓÃ·¨is {}
%centroids = cellfun(@LocateWormCentre,CC.PixelIdxList);
centroids = cellfun(@LocateWormBodyMiddle,CC.PixelIdxList);
s = size(imgremoved);
[x,y] = ind2sub(s,centroids);
imgx = y;
imgy = x;
%cat(1,imgx(1:10),imgy(1:10))
%imgx(1:10);
%imgy(1:10);
%class(imgremoved)
%centroids = regionprops(imgremoved,'centroid');
%centroids = cat(1,centroids.Centroid);
%plot(centroids(:,1),centroids(:,2),'*')
%imgremoved = imrotate(imgremove,180);
%figure(1) 
%imshow(imgremoved)
figure(2)
%imshow(imgremoved(10:100,10:100))
imshow(imcrop(imgremoved,[1000 1 2000 2000]))
hold on
title(imgename)
%plot(imgx,imgy,'r*','MarkerSize',1)
plot(imgx-1000,imgy-1,'r*','MarkerSize',1)
%hold on
%plot([2372.0],[s(2)-724.2],'b*','MarkerSize',1)
%hold on
%plot([2398.0],[s(2)-669.8],'g*','MarkerSize',1)
imgy = s(2)-imgy;
%hold on
%plot(987,500,'*','MarkerSize',10)
%hold on
%plot(1198,500,'*','MarkerSize',10)
class(imgremoved)
imwrite(imgremoved,...
    ['\\10.20.13.222\igem\WormTrack\WormTrack_summer\ImgeProcessing\' regexprep(imgename,'fit','png' )],'png')
%imwrite(gcf,['\\10.20.13.222\igem\WormTrack\WormTrack_summer\' imgename],'tif')
hold off
%figure(5)
%plot(imgx,s(2)-imgy,'r*','MarkerSize',1)