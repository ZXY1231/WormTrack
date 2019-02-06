clear;
x1=[1,2];
y1=[3,4];

x2=[3];
y2=[2];
%NextFrameNearestPoint(x1,y1,x2,y2);
a = cat(1,x1,y1)
cat(1,a,[5 6])
cat(2,x1,y1)
a(1,:)

%{
i = linspace(1,10,10);
i = 1:10;
flocates = fopen('Locations.txt','w');
fprintf(flocates,'test\r\n');
fprintf(flocates,'%f  ',x1);
fprintf(flocates,'\r\n');
fprintf(flocates,'%f  ',y1);
%}


