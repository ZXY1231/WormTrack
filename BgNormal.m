function contrastAdjusted = BgNormal(img)
%img = imread('/Users/apple/Desktop/IGEM/post/tracking/WormTrack/worm1.bmp');
%img = img-4000;
%max(img)
%min(img)
%function 
se = offsetstrel('ball',50,50);
%img = imadjust(img);
img = padarray(img,[8,8],'symmetric');
img = wiener2(img,[9,9]);

%imgopening = imtophat(imgwiener,se);

tophatFiltered = imtophat(img,se);
%figure(1) 
%imshow(img);

%%figure(2)
%%imshow(tophatFiltered);

contrastAdjusted = imadjust(tophatFiltered);
figure(3)
imshow(contrastAdjusted);
%max(contrastAdjusted)
%min(contrastAdjusted)
%imwrite(contrastAdjusted,'worm1_contrastAdjusted.tif')

%BothatFiltered = imbothat(img,se);
%figure(4)
%imshow(BothatFiltered);

%figure(5)
%imshow(imadjust(BothatFiltered));

