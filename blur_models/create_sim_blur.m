clear all
close all
clc

load sim.mat
[nx,ny] = size(s);

for i=1:8
tit = ['blur_',int2str(i),'.mat'];
load(tit)
B = psf2otf(blur,[nx,ny])

figure
%subplot(131)
imagesc(s); axis image off;%colormap(gray)
print -dpng original
%subplot(132)
imagesc(real(ifft2(B.*fft2(s)))); axis image off;%colormap(gray)
print([tit(1:end-4),'_degraded'],'-dpng')
%subplot(133)
imagesc(blur); axis image off;%colormap(gray)
print([tit(1:end-4)],'-dpng')
end
