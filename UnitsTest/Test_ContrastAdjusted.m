clear all;
img1 = imread('C.elegans_203550_0033.tif');
contrastAdjusted1 = BgNormal(img1);
imwrite(contrastAdjusted1, 'C.elegans_203550_0033_clip_winner_3_3rolling_adjusted.tif');
%figure(2);
%imshow(contrastAdjusted1);    
BgThresh(contrastAdjusted1,'C.elegans_203550_0033_clip_winner_3_3rolling_adjusted_thresh.tif');




%img2 = imread('C.elegans_235855_0000.fit');
%contrastAdjusted2 = BgNormal(img2);
%imwrite(contrastAdjusted2, 'C.elegans_235855_0000_contrast.tif');
%figure(3);
%imshow(contrastAdjusted2)

