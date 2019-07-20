function location = LocateWormCentre(idx,s)
% Return linear index of centroid  
% | Version | Author | Date     | Commit
% | 0.1     | ZhouXY | 18.07.19 | The init version

% Calculate worm centre
% img is a n*m mtricx
n = s(1);
[x,y] = ind2sub(s,idx);
leng = length(idx);
location = round(sum(x))/leng + n*round(sum(y)/leng-1);
% pay attention to round(), it's critical
