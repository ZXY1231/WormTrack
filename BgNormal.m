function contrastAdjusted = BgNormal(img)
% Backgroud Normalize
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version
% | 0.2     | H.F.   | 18.09.05 |
% To Do: using SNR extend instead of imadjust

% Before the calculation, the image can be down dimension to acclation.
se = offsetstrel('ball',50,50);
%img = imadjust(img);
%img = padarray(img,[8,8],'symmetric');
img = wiener2(img,[3,3]);

%imgopening = imtophat(imgwiener,se);
%tic;
tophatFiltered = imtophat(img,se);
%contrastAdjusted =tophatFiltered;
%toc;

%%imshow(tophatFiltered);
contrastAdjusted = imadjust(tophatFiltered);

