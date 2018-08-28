% Main function to extra features from worm image
%
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version
% | 0.2     | H.F.   | 18.08.28 | 
% TODO:

clear all;
warning('off','all'); % Close all warning

tic;
% Parameter
%path = '/home/hf/iGEM/DataII/20180824/C.elegans/240818_115847_130_During1hr/';
path = '/home/hf/iGEM/DataI/20180819/C.elegans/190818_203550_130_Alcohol_Starting/';
%path = '\\10.20.13.222\igem\Results\20180709\130Worm-First30min\';

imges = dir([path '*.tif']);
imge_num = length(imges);
CentroidsLocates = cell(imge_num,1);
%leng = imge_num;
leng = 20; % debug

% Handling background and worms localization
parfor i =1:leng
 %parfor i =40:leng % debug
    img = imread([imges(i).folder '/' imges(i).name]);
    img = img(60:3450, 141:3531);
    contrastAdjusted = BgNormal(img);
    imwrite(img, ['Normalize/' imges(i).name '.png']);
    [imgx,imgy] = BgThresh(contrastAdjusted,imges(i).name);
    CentroidsLocates{i,1} = cat(1,imgx,imgy);
    %printf('Remain %d', leng-i);
end

% The worm location in first frame
PreLocaX = CentroidsLocates{1,1}(1,:);
PreLocaY = CentroidsLocates{1,1}(2,:);

% Postions of each worm
PointsDynamics = cell(length(PreLocaX),1);
for i = 1:length(PreLocaX)
    % Save each worm's location in frist frame
    PointsDynamics{i,1} = [PreLocaX(i) PreLocaY(i)];
end

% 
for i = 2:leng 
    PostLoca = CentroidsLocates{i,1};
    [NeX,NeY] = NextFrameNearestPoint(PreLocaX,PreLocaY,PostLoca(1,:),PostLoca(2,:));
    PreLocaX = NeX;
    PreLocaY = NeY;
    % Each worm 
    for j = 1:length(PreLocaX)
        % Stack the every point
        PointsDynamics{j,1} = cat(1,PointsDynamics{j,1},[PreLocaX(j) PreLocaY(j)]);
    end
    %fprintf(flocates,'%d\n%lf\n%lf\n\n',i,PreLocaX,PreLocaY);
end

% Write each worm's localtion to file
flocates = fopen('/home/hf/iGEM/Results/20180819/190818_203550_130_Alcohol_Starting/location_0819','w');
for i = 1:length(PointsDynamics)
    fprintf(flocates, 'Point %d', i);
    fprintf(flocates, '\r\nx\r\n');
    fprintf(flocates, '%f\r\t', PointsDynamics{i,1}(:,1));
    fprintf(flocates, '\r\ny\r\n');
    fprintf(flocates, '%f\r\t', PointsDynamics{i,1}(:,2));
    fprintf(flocates, '\r\n\r\n');
end
toc;