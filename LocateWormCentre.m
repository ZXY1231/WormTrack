function location = LocateWormCentre(idx)%centroid for linear 
%s = [3268 3272];
%s = [3298 3460];
%s = [3314 3476];
%s = [3284 3288];
%s = [976 1296];
s = [1030 1040];
%img is a n*m mtricx
n=s(1);
%{
idx;
leng = length(idx);
x = floor((idx-1)/n)+1;
%y = n-(mod(idx-1,n)+1)+1;
y = n-mod(idx-1,n);
x
(sum(x)/leng)
%}
[x,y] = ind2sub(s,idx);
%x
leng = length(idx);
location = round(sum(x))/leng+n*round(sum(y)/leng-1);%pay attention to round(), it's critical
%xsub = round(sum(x)/leng)
%ysub = round(sum(y)/leng)
%location = sub2ind(s,xsub,ysub)
