img = imread('\\10.20.13.222\igem\WormTrack\Puncta_summer\3_hela_LC3_sc1_ct_G0.6_R4_01_ch00.tif');
%img = imread('\\10.20.13.222\igem\Results\20180709\130Worm-First30min\C.elegans_235855_0000.fit');
%size(img)
%Handling background and worms localization
contrastAdjusted = BgNormal(img);
[imgx,imgy] = BgThresh(contrastAdjusted,'test');