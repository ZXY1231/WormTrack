%{
a =[1,0,0,1];
b = [0,1,1,0];
ball = [0,1,0;1,0,1;0,1,0]
balllinearindex = {2,4,6,8}
img = imread('sphere.tif');
CC = bwconncomp(img,8)
centroids = cellfun(@LocateWormCentre,CC.PixelIdxList)
[x,y] = ind2sub([3,3],centroids)
%centroids = regionprops(ball,'centroid');
%centroids.Centroid;
%LocateWormCentre(balllinearindex)
mod(1059203.2170999998,976)
%imwrite([[0 1 0];[1 0 1];[0 1 0]],'\\10.20.13.222\igem\WormTrack\WormTrack_summer\test.tif','tif')
a = [1,2,3;1,2,3]
sum(a)
sum(a')
%}
b = [0,2,0;1,0,3;0,2,0]
c = logical(b)
%test = imread('\\10.20.13.222\igem\WormTrack\WormTrack_summer\worm1.bmp');
c = c
figure(1)
class(c)
imwrite(c,'\\10.20.13.222\igem\WormTrack\WormTrack_summer\test.tif','tif')









