function [NearestX,NearestY] = NextFrameNearestPoint(PreX,PreY,PostX,PostY)
% Return worm location in next frame by comparing the distance
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version

Preleng = length(PreX);
Postleng = length(PostX);
%Nearest = zeros(leng,2);
NearestX = zeros(Preleng,1);
NearestY = zeros(Preleng,1);

% Find out same worm in two frame
for i = linspace(1, Preleng, Preleng) % load worm in next frame
    x = PreX(i);
    y = PreY(i);
    near = 9999999;
    for j = linspace(1, Postleng, Postleng) %load worm in post frame
      dis = (x-PostX(j))^2 + (y-PostY(j))^2;
      if dis < near
          near = dis;
          NeX = PostX(j);
          NeY = PostY(j);
      end
    end
    NearestX(i) = NeX;
    NearestY(i) = NeY;
end
