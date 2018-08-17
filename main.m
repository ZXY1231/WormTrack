%-------------------------------------------------------------------------%
% Description: Main function to extra features from worm image
% | Version | Author | Date     | Commit 
% | 0.1     | ZhouXY | 18.07.19 | The init version

%path = '/home/hf/iGEM/Results/20180709/130Worm-First30min/';
path = '\\10.20.13.222\igem\Results\20180709\130Worm-First30min\';
imges = dir([path '*.fit']);
imge_num = length(imges);
CentroidsLocates = cell(imge_num,1);
leng = imge_num-10;
%leng = 10;
for i =1:leng
    %[imges(i).folder '\' imges(i).name];
    img = imread([imges(i).folder '/' imges(i).name]);
    %size(img)
    %Handling background and worms localization
    contrastAdjusted = BgNormal(img);
    [imgx,imgy] = BgThresh(contrastAdjusted,imges(i).name);
    CentroidsLocates{i,1} = cat(1,imgx,imgy);
end
PreLocaX = CentroidsLocates{1,1}(1,:);
PreLocaY = CentroidsLocates{1,1}(2,:);

%figure(2)
%subplot(4,4,1)
%plot(PreLocaX,PreLocaY,'*','MarkerSize',1);
%figure(3)
%subplot(4,4,1)
%plot(PreLocaX,PreLocaY,'*','MarkerSize',1);
%figure(3)
%h = plot(PreLocaX,PreLocaY,'*','MarkerSize',1);

PointsDynamics = cell(length(PreLocaX),1);
for i = 1:length(PreLocaX)
    PointsDynamics{i,1} = [PreLocaX(i) PreLocaY(i)];
end
PointsDynamics{1,1}
PointsDynamics{2,1}
for i = 2:leng
    i;
    PostLoca = CentroidsLocates{i,1};
    [NeX,NeY] = NextFrameNearestPoint(PreLocaX,PreLocaY,PostLoca(1,:),PostLoca(2,:));
    %set(h,'XData',PreLocaX,'YData',PreLocaY,'MarkerSize',1);
    %drawnow
    %figure(2);
    %subplot(4,4,i);
    %plot(PreLocaX,PreLocaY,'*','MarkerSize',1);
    %figure(3);
    %subplot(4,4,i)
    %plot(PostLoca(1,:),PostLoca(2,:),'*','MarkerSize',1); 
    PreLocaX = NeX;
    PreLocaY = NeY;
    for j = 1:length(PreLocaX)
        PointsDynamics{j,1} = cat(1,PointsDynamics{j,1},[PreLocaX(j) PreLocaY(j)]);
    end
    %fprintf(flocates,'%d\n%lf\n%lf\n\n',i,PreLocaX,PreLocaY);
end
PointsDynamics{1,1}
PointsDynamics{2,1}

flocates = fopen('Locations.txt','w');
length(PointsDynamics)
for i = 1:length(PointsDynamics)
    i;
    fprintf(flocates,'Point %d',i);
    fprintf(flocates,'\r\nx\r\n');
    fprintf(flocates,'%f\r\t',PointsDynamics{i,1}(:,1));
    fprintf(flocates,'\r\ny\r\n');
    fprintf(flocates,'%f\r\t',PointsDynamics{i,1}(:,2));
    fprintf(flocates,'\r\n\r\n');
end


%{
img = imread('C.elegans_003323_0001.tif');
%Handling Background and Worms Localization
contrastAdjusted = BgNormal(img);
[imgx,imgy] = BgThresh(contrastAdjusted);
%}
