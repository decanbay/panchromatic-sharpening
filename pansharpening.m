%%
clear all close all; clc
%% Diskteki resme pan keskinle�tirme yapmak i�in bu k�s�m 
dosya_ms='istanbulMS.jpg';
dosya_pan='istanbulPAN.jpg';
ms=im2double(imread(dosya_ms));  % MS resmin diskten okunup double precision a �evirilmesi.
pan=im2double(imread(dosya_pan)); % Pankromatik resim i�in ayn� i�lem

%% mat dsyas�ndaki resme keskinle�tirme yapmak i�in bu k�s�m

% load('image2M.mat');    % bu 4 sat�r�n ba��ndaki  %leri kald�r�n,
% load('image2P.mat');    % dosyadan okumak i�in ise buradaki 4 sat�rda da % olmal�
% ms=im2double(A); 
% pan=im2double(B);
% dosya_pan='image2pan.jpg';
%% ///////////////////  IHS Metodu  \\\\\\\\\\\\\\\\\ %%

I0 = ms(:,:,1)*1/3 + ms(:,:,2)*1/3 + ms(:,:,3)*1/3; % burada MS resmin Intensity bile�eni elde ediliyor (IHS d�n���m�)
Rnew=ms(:,:,1) + ( pan - I0 ); % Burada da k�rm�z� bant i�in pan keskinle�tirme yap�l�yor
Gnew=ms(:,:,2) + ( pan - I0 );%ye�il bant
Bnew=ms(:,:,3) + ( pan - I0 );% mavi bant i�in ayn� �ekilde
IHS_keskin=zeros([size(pan),3]); % keskinle�tirilmi� resmi tutacak matris olu�turulmas�
IHS_keskin(:,:,1)=Rnew; % keskinle�tirilmi� matrisin k�rm�z� kanal�na e�itlenmesi
IHS_keskin(:,:,2)=Gnew;%ye�il kanal
IHS_keskin(:,:,3)=Bnew;% mavi kanal e�itlemesi
% bu i�lemden sonra sharpened matrisi bizim keskinle�tirilmi� resmimizi RGB
% resim olarak tutuyor, imshow fonkssiyonu ile g�r�nt�lenecek olunursa
figure
imshow(ms(:,:,1:3)) % multispectral resmi g�r�nt�lemek i�in
figure
imshow(IHS_keskin);  % keskinle�tirilmi� resmin g�r�nt�lenmesi
figure
imshow(pan)% PANkromatik resmin g�r�nt�lenmesi 
imwrite(IHS_keskin,strcat('IHS_',dosya_pan)); % bu resmi diske JPG format�nda kaydetmek i�in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  Brovey Metodu  \\\\\\\\\\\\\\\\\ %%
ms_toplam=zeros(size(ms,1),size(ms,2));
Brovey_keskin=zeros(size(ms,1),size(ms,2),3);

for ind=1:3
ms_toplam=ms_toplam+ms(:,:,ind);
end
ms_toplam=ms_toplam/size(ms,3); % parlakl���n tekrar d���r�lmesi

Brovey_R=ms(:,:,1).*pan./ms_toplam;
Brovey_G=ms(:,:,2).*pan./ms_toplam;
Brovey_B=ms(:,:,3).*pan./ms_toplam;
Brovey_keskin(:,:,1)=Brovey_R;
Brovey_keskin(:,:,2)=Brovey_G;
Brovey_keskin(:,:,3)=Brovey_B;
figure
imshow(Brovey_keskin)
imwrite(Brovey_keskin,strcat('Brovey_',dosya_pan)); % bu resmi diske JPG format�nda kaydetmek i�in

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  PCA Metodu  \\\\\\\\\\\\\\\\\ %%

X=reshape(ms,[],size(ms,3));
[height,width,depth] = size(ms); 
mx = mean(X); 
N = size(X,1);
coeff=pca(X); % PCA coefficientlar� ayn� zamanda eigenvekt�rlerin b�y�kten k����e s�ral� hali
Y=coeff'*(X - repmat(mx,N,1))'; % burada repmat mx ile ortalama de�er x ten ��kar�larak resimdeki DC komponent yok ediliyor.
Y1 = reshape(Y(1,:), height, width); %1. PC
Y2 = reshape(Y(2,:), height, width); % 2. PC
Y3 = reshape(Y(3,:), height, width); %3. PC
figure;
subplot(1,3,1), imshow(Y1,[]);
subplot(1,3,2), imshow(Y2,[]);
title '1. 2. ve 3. Principal Component'
subplot(1,3,3), imshow(Y3,[]);

pan_v=reshape(pan,[],1);% pan resmin vekt�r format�na getirilmesi
X_pan = ( coeff(:,2:size(ms,3)) * Y(2:size(ms,3),:) )' + repmat(pan_v,1,size(ms,3)); 
PCA_keskin=zeros(size(ms,1),size(ms,2),3);
PCA_keskin(:,:,1) = reshape(X_pan(:,1), height, width);
PCA_keskin(:,:,2) = reshape(X_pan(:,2), height, width);
PCA_keskin(:,:,3) = reshape(X_pan(:,3), height, width);
figure
imshow(PCA_keskin)
imwrite(PCA_keskin,strcat('PCA_',dosya_pan));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  GS Metodu  \\\\\\\\\\\\\\\\\ %%

[height,width,depth] = size(ms);
toplam_ms=zeros(size(ms,1),size(ms,2));
for ind=1:size(ms,3)
    toplam_ms =  toplam_ms + ms(:,:,ind); %multispectral bantlar�n toplam�
end
ort_ms= toplam_ms/size(ms,3);%ms bantlar�n ortalamas�(simulated panchromatic band)
U=[reshape(ort_ms,[],1) reshape(ms,[],size(ms,3))]; % U=V*C ;C upper triangular  
figure; imshow(ms(:,:,1:3))
figure; imshow(ort_ms)
[V,C]=gsog(U); % Gram schmidt ortogonalle�tirme fonksiyonu
V_mod=V; % g�ncellenecek olan vekt�rler ilk olarak ayn�sna e�i�tlenip sonra
V_mod(:,1)=reshape(pan,[],1); %ilk de�er pan resim ile de�i�tiriliyor
U_mod=V_mod*C;% ters GS ile de�i�tirilmi� U elde edilip buradan keskinle�tirilmi�
                % resimler �ekiliyor
r=reshape(U_mod(:,2),height, width); %tekrar vekt�rden resim �ekline matris d�zenlemesi
g=reshape(U_mod(:,3),height, width); % ye�il ve mavi kanal i�in de benzer.
b=reshape(U_mod(:,4),height, width);
GS_keskin=zeros([size(pan) 3]); % keskinle�tirilmi� resim
GS_keskin(:,:,1)=r;
GS_keskin(:,:,2)=g;
GS_keskin(:,:,3)=b;
figure
imshow(GS_keskin)
imwrite(GS_keskin,strcat('GS_',dosya_pan))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  Wavelet Metodu  \\\\\\\\\\\\\\\\\ %%
% clear all
% clc
% dosya_ms='istanbulMS.jpg';
% dosya_pan='istanbulPAN.jpg';
% ms=im2double(imread(dosya_ms));  % MS resmin diskten okunup double precision a �evirilmesi.
% pan=im2double(imread(dosya_pan)); % Pankromatik resim i�in ayn� i�lem
[height,width,depth] = size(ms);
for level=1:3 % wave d�n���m�n�n seviyesi

waveletname='haar'; % 'https://www.mathworks.com/help/wavelet/ref/wfilters.html  
[panvec, s] = wavedec2(pan,level,waveletname); %pan resmin wavelet d�n���m� 
% bu linke bak�n�z   https://www.mathworks.com/help/wavelet/ref/wavedec2.html
for i=1:depth
    reconstvec(:,i) = panvec; % pankromatik resmin y�ksek ��z�n�rl�k bilgisinin pan resimden al�nmas�
    ms_dec(:,i) = wavedec2(ms(:,:,i),level,waveletname); % ms resmin wavelet d�n���m� (t�m bantlar i�in)
end

for j=1:s(1,1) * s(1,2)
        reconstvec(j,:) = ms_dec(j,:); % pankromatik resmin d���k ��z�n�rl�kl� bilgisinin ms den al�nmas�
end

%reconstruct image doing inverse wavelet transform
for i=1:depth
    Wave_keskin(:,:,i) = waverec2(reconstvec(:,i),s,waveletname); % ters wavelet d�n���m� ile son resmin elede edilmesi
end
imshow(Wave_keskin(:,:,1:3))
imwrite(Wave_keskin(:,:,1:3),strcat('Wave_','level_',num2str(level),'_',waveletname,'_',dosya_pan));
clear panvec s reconstvec  ms_dec
end

%% YUV colorspace
y=rgb2ycbcr(ms);
y2=zeros(size(y));
y2(:,:,1)=pan;
y2(:,:,2:3)=y(:,:,2:3);
y3=ycbcr2rgb(y2);
figure
imshow(y3,[])
imwrite(y3,strcat('YUV_',dosya_pan))

figure
imshow(Brovey_keskin,[])
[hata,~]=Metric_RMSE(ms,PCA_keskin)
Metric_ERGAS(ms,PCA_keskin)
Metric_ERGAS(ms,y3)
Metric_Results_yuv_1 =  Metrics( ms, y3 , pan )
%% unsharp mask yuv
yy=rgb2ycbcr(ms);
yy2=zeros(size(yy));
yy2(:,:,1)=pan;
yy2(:,:,2)=imsharpen(yy(:,:,2),'Radius',2,'Amount',0.9);%,'Threshold',0.7);
yy2(:,:,3)=imsharpen(yy(:,:,3),'Radius',2,'Amount',0.9);%,'Threshold',0.7);
yy3=ycbcr2rgb(yy2);
yy4=rgb2ycbcr(yy3);
yy5=zeros(size(yy));
yy5(:,:,1)=pan;
yy5(:,:,2)=yy4(:,:,2);
yy5(:,:,3)=yy4(:,:,3);
yy6=ycbcr2rgb(yy5);
imshow(yy3)
Metric_Results_yuv_sharp =  real(Metrics( ms, yy3 , pan ));
Metric_Results_yuv_sharp2 =  Metrics( ms, yy6 , pan );

[Metric_Results_yuv_1;Metric_Results_yuv_sharp;Metric_Results_yuv_sharp2; Metrics( ms, yy6 , pan )]
%% LAB
y=rgb2lab(ms);
y2=zeros(size(x));
y2(:,:,1)=pan;
y2(:,:,2:3)=y(:,:,2:3);
y3=lab2rgb(y2);
figure
imshow(y3,[])
figure
imshow(Brovey_keskin,[])
%% HSV
yhsv=rgb2hsv(ms);
y2hsv=zeros(size(y));
y2hsv(:,:,3)=pan;
y2hsv(:,:,1:2)=yhsv(:,:,1:2);
y3hsv=hsv2rgb(y2hsv);
figure
imshow(y3hsv,[])
figure
imshow(Brovey_keskin,[])
%% YUV-PCA
X2=reshape(pan,[],size(pan,3));
[height2,width2,depth2] = size(pan); 
mx2 = mean(X2); 
N2 = size(X2,1);
coeff2=pca(X2); % PCA coefficientlar� ayn� zamanda eigenvekt�rlerin b�y�kten k����e s�ral� hali
Y2=coeff2'*(X2 - repmat(mx2,N2,1))'; % burada repmat mx ile ortalama de�er x ten ��kar�larak resimdeki DC komponent yok ediliyor.
Y3 = reshape(Y2(1,:), height, width); %1. PC pan
% Y31=
Y4=zeros(size(ms));
Y4(:,:,1)=Y3; 
Y4(:,:,2:3)=yy(:,:,2:3);
Y5=ycbcr2rgb(Y4);
imshow(Y5,[])
[Metric_Results_yuv_1;Metric_Results_yuv_sharp;Metric_Results_yuv_sharp2; Metrics( ms, Y5 , pan )]
%%
qq=fft2((pan));
qqq=reshape(qq,[],size(qq,3));
[height2,width2,depth2] = size(qq); 
mx2 = mean(qqq); 
N2 = size(qqq,1);
coeff2=pca(qqq); % PCA coefficientlar� ayn� zamanda eigenvekt�rlerin b�y�kten k����e s�ral� hali
Y2=coeff2'*(qqq - repmat(mx2,N2,1))'; % burada repmat mx ile ortalama de�er x ten ��kar�larak resimdeki DC komponent yok ediliyor.
Y3 = reshape(Y2(1,:), height, width); %1. PC pan
Y4=ifft2(Y3);
Y5=imrotate(Y4,180);
imshow(imrotate(Y4,180));

Y6=zeros(size(ms));
Y6(:,:,1)=real(Y5);
Y6(:,:,2:3)=yy(:,:,2:3);  
Y7=ycbcr2rgb(Y6);
imshow(Y7,[])
entropies=[entropy(IHS_keskin); entropy(Brovey_keskin); entropy(PCA_keskin); entropy(Wave_keskin); entropy(GS_keskin);entropy(yy3)] 
[Metric_Results_yuv_1;Metric_Results_yuv_sharp;Metric_Results_yuv_sharp2; Metrics( ms, Y7 , pan )]