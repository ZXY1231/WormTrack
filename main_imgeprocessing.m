%-------------------------------------------------------------------------%
% Description: Main function to extra features from worm image
% | Version | Author | Date     | Commit 
% | 0.1     | ZhouXY | 18.07.19 | The init version

%path = '/home/hf/iGEM/Results/20180709/130Worm-First30min/';
path = '\\10.20.13.222\igem\Results\20180709\130Worm-First30min\';
imges = dir([path '*.fit']);
imge_num = length(imges);cccc
CentroidsLocates = cell(imge_num,1);
leng = imge_num;
%leng = 1;
for i =1:leng
    %[imges(i).folder '\' imges(i).name];
    img = imread([imges(i).folder '/' imges(i).name]);
    %size(img)
    %Handling background and8 worms localization
    contrastAdjusted = BgNormal(img);
    [imgx,imgy] = BgThresh(contrastAdjusted,imges(i).name);
    %display([max(imgx),max(imgy),min(imgx),min(imgy)])
    %CentroidsLocates{i,1} = cat(1,imgx,imgy);
end