function contrastAdjusted = BgNormal(img)
% Backgroud Normalize

% Before the calculation, the image can be down dimension to acclation.
se = offsetstrel('ball',50,50);
%img = imadjust(img);
%img = padarray(img,[8,8],'symmetric');
img = wiener2(img,[3,3]);

%imgopening = imtophat(imgwiener,se);
%tic;
tophatFiltered = imtophat(img,se);
%toc;

%%imshow(tophatFiltered);
contrastAdjusted = imadjust(tophatFiltered);
