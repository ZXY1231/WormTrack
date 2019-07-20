% Main function to extra features from worm image
%
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version
% | 0.2     | H.F.   | 18.08.28 | 
% TODO: put the parameters outside function but main function
% å?¯å?‘å¼?è¿½è¸ª
clear all;
warning('off','all'); % Close all warning

tic;
% Parameters
source_path = '/home/hf/iGEM/Results/20180904/DifussionTest7min/';
result_path ='/home/hf/iGEM/Results/20180904/';
thresh = 0.991;
small = 50;
big = 600;

imges = dir([source_path '*.tif']);
imge_num = length(imges);
CentroidsLocates = cell(imge_num,1);
leng = imge_num;
%leng = 16 ; %debug

% Handling background and worms localization
parfor i =1:leng
    img = imread([imges(i).folder '/' imges(i).name]);
    img = img(516:1978, 588:1982); 
    
    contrastAdjusted = BgNormal(img);
    imwrite(img, [result_path '/Normalize/' imges(i).name '.png']);
    

    [imgx, imgy, imgbwthresh, imgremoved] = BgThresh(contrastAdjusted,thresh,small,big);
    
    imwrite(imgbwthresh, [result_path 'Thresh/' imges(i).name '.png']);
    imwrite(imgremoved, [result_path 'Remove/' imges(i).name '.png']);

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
flocates = fopen([result_path 'DiffussionTest7min.localtion'],'w');
csvwrite([result_path 'DiffussionTest7min.local'],cell2mat(PointsDynamics'));
for i = 1:length(PointsDynamics)
    fprintf(flocates, 'Point %d', i);
    fprintf(flocates, '\r\nx\r\n');
    fprintf(flocates, '%f\r\t', PointsDynamics{i,1}(:,1));
    fprintf(flocates, '\r\ny\r\n');
    fprintf(flocates, '%f\r\t', PointsDynamics{i,1}(:,2));
    fprintf(flocates, '\r\n\r\n');
end
toc;
