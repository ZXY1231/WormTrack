img1 = imread('C.elegans_203550_0033.tif');
contrastAdjusted1 = gather( BgNormal(gpuArray(img1)));
imwrite(contrastAdjusted, 'C.elegans_203550_0033_contrast.tif');
%figure(2);
imshow(gather(contrastAdjusted1));

%img2 = fitsread('C.elegans_235855_0000.fit');
%contrastAdjusted2 = gather(BgNormal(gpuArray(img2)));
%imwrite(contrastAdjusted2, 'C.elegans_235855_0000_contrast.tif');
%figure(3);
%imshow(contrastAdjusted2)