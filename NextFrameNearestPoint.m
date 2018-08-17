function [NearestX,NearestY] = NextFrameNearestPoint(PreX,PreY,PostX,PostY)
Preleng = length(PreX);
Postleng = length(PostX);
%Nearest = zeros(leng,2);
NearestX = zeros(Preleng,1);
NearestY = zeros(Preleng,1);
for i = linspace(1,Preleng,Preleng)
    x = PreX(i);
    y = PreY(i);
    near = 9999999;
    for j = linspace(1,Postleng,Postleng)
      dis = (x-PostX(j))^2+(y-PostY(j))^2;
      if dis<near
          near = dis;
          NeX = PostX(j);
          NeY = PostY(j);
      end
    end
    NearestX(i) = NeX;
    NearestY(i) = NeY;
end
NearestX;
NearestY;
