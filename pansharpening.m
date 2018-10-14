%%
clear all close all; clc
%% Diskteki resme pan keskinleþtirme yapmak için bu kýsým 
dosya_ms='istanbulMS.jpg';
dosya_pan='istanbulPAN.jpg';
ms=im2double(imread(dosya_ms));  % MS resmin diskten okunup double precision a çevirilmesi.
pan=im2double(imread(dosya_pan)); % Pankromatik resim için ayný iþlem

%% mat dsyasýndaki resme keskinleþtirme yapmak için bu kýsým

% load('image2M.mat');    % bu 4 satýrýn baþýndaki  %leri kaldýrýn,
% load('image2P.mat');    % dosyadan okumak için ise buradaki 4 satýrda da % olmalý
% ms=im2double(A); 
% pan=im2double(B);
% dosya_pan='image2pan.jpg';
%% ///////////////////  IHS Metodu  \\\\\\\\\\\\\\\\\ %%

I0 = ms(:,:,1)*1/3 + ms(:,:,2)*1/3 + ms(:,:,3)*1/3; % burada MS resmin Intensity bileþeni elde ediliyor (IHS dönüþümü)
Rnew=ms(:,:,1) + ( pan - I0 ); % Burada da kýrmýzý bant için pan keskinleþtirme yapýlýyor
Gnew=ms(:,:,2) + ( pan - I0 );%yeþil bant
Bnew=ms(:,:,3) + ( pan - I0 );% mavi bant için ayný þekilde
IHS_keskin=zeros([size(pan),3]); % keskinleþtirilmiþ resmi tutacak matris oluþturulmasý
IHS_keskin(:,:,1)=Rnew; % keskinleþtirilmiþ matrisin kýrmýzý kanalýna eþitlenmesi
IHS_keskin(:,:,2)=Gnew;%yeþil kanal
IHS_keskin(:,:,3)=Bnew;% mavi kanal eþitlemesi
% bu iþlemden sonra sharpened matrisi bizim keskinleþtirilmiþ resmimizi RGB
% resim olarak tutuyor, imshow fonkssiyonu ile görüntülenecek olunursa
figure
imshow(ms(:,:,1:3)) % multispectral resmi görüntülemek için
figure
imshow(IHS_keskin);  % keskinleþtirilmiþ resmin görüntülenmesi
figure
imshow(pan)% PANkromatik resmin görüntülenmesi 
imwrite(IHS_keskin,strcat('IHS_',dosya_pan)); % bu resmi diske JPG formatýnda kaydetmek için
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  Brovey Metodu  \\\\\\\\\\\\\\\\\ %%
ms_toplam=zeros(size(ms,1),size(ms,2));
Brovey_keskin=zeros(size(ms,1),size(ms,2),3);

for ind=1:3
ms_toplam=ms_toplam+ms(:,:,ind);
end
ms_toplam=ms_toplam/size(ms,3); % parlaklýðýn tekrar düþürülmesi

Brovey_R=ms(:,:,1).*pan./ms_toplam;
Brovey_G=ms(:,:,2).*pan./ms_toplam;
Brovey_B=ms(:,:,3).*pan./ms_toplam;
Brovey_keskin(:,:,1)=Brovey_R;
Brovey_keskin(:,:,2)=Brovey_G;
Brovey_keskin(:,:,3)=Brovey_B;
figure
imshow(Brovey_keskin)
imwrite(Brovey_keskin,strcat('Brovey_',dosya_pan)); % bu resmi diske JPG formatýnda kaydetmek için

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ///////////////////  PCA Metodu  \\\\\\\\\\\\\\\\\ %%

X=reshape(ms,[],size(ms,3));
[height,width,depth] = size(ms); 
mx = mean(X); 
N = size(X,1);
coeff=pca(X); % PCA coefficientlarý ayný zamanda eigenvektörlerin büyükten küçüðe sýralý hali
Y=coeff'*(X - repmat(mx,N,1))'; % burada repmat mx ile ortalama deðer x ten çýkarýlarak resimdeki DC komponent yok ediliyor.
Y1 = reshape(Y(1,:), height, width); %1. PC
Y2 = reshape(Y(2,:), height, width); % 2. PC
Y3 = reshape(Y(3,:), height, width); %3. PC
figure;
subplot(1,3,1), imshow(Y1,[]);
subplot(1,3,2), imshow(Y2,[]);
title '1. 2. ve 3. Principal Component'
subplot(1,3,3), imshow(Y3,[]);

pan_v=reshape(pan,[],1);% pan resmin vektör formatýna getirilmesi
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
    toplam_ms =  toplam_ms + ms(:,:,ind); %multispectral bantlarýn toplamý
end
ort_ms= toplam_ms/size(ms,3);%ms bantlarýn ortalamasý(simulated panchromatic band)
U=[reshape(ort_ms,[],1) reshape(ms,[],size(ms,3))]; % U=V*C ;C upper triangular  
figure; imshow(ms(:,:,1:3))
figure; imshow(ort_ms)
[V,C]=gsog(U); % Gram schmidt ortogonalleþtirme fonksiyonu
V_mod=V; % güncellenecek olan vektörler ilk olarak aynýsna eþiþtlenip sonra
V_mod(:,1)=reshape(pan,[],1); %ilk deðer pan resim ile deðiþtiriliyor
U_mod=V_mod*C;% ters GS ile deðiþtirilmiþ U elde edilip buradan keskinleþtirilmiþ
                % resimler çekiliyor
r=reshape(U_mod(:,2),height, width); %tekrar vektörden resim þekline matris düzenlemesi
g=reshape(U_mod(:,3),height, width); % yeþil ve mavi kanal için de benzer.
b=reshape(U_mod(:,4),height, width);
GS_keskin=zeros([size(pan) 3]); % keskinleþtirilmiþ resim
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
% ms=im2double(imread(dosya_ms));  % MS resmin diskten okunup double precision a çevirilmesi.
% pan=im2double(imread(dosya_pan)); % Pankromatik resim için ayný iþlem
[height,width,depth] = size(ms);
for level=1:3 % wave dönüþümünün seviyesi

waveletname='haar'; % 'https://www.mathworks.com/help/wavelet/ref/wfilters.html  
[panvec, s] = wavedec2(pan,level,waveletname); %pan resmin wavelet dönüþümü 
% bu linke bakýnýz   https://www.mathworks.com/help/wavelet/ref/wavedec2.html
for i=1:depth
    reconstvec(:,i) = panvec; % pankromatik resmin yüksek çözünürlük bilgisinin pan resimden alýnmasý
    ms_dec(:,i) = wavedec2(ms(:,:,i),level,waveletname); % ms resmin wavelet dönüþümü (tüm bantlar için)
end

for j=1:s(1,1) * s(1,2)
        reconstvec(j,:) = ms_dec(j,:); % pankromatik resmin düþük çözünürlüklü bilgisinin ms den alýnmasý
end

%reconstruct image doing inverse wavelet transform
for i=1:depth
    Wave_keskin(:,:,i) = waverec2(reconstvec(:,i),s,waveletname); % ters wavelet dönüþümü ile son resmin elede edilmesi
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
coeff2=pca(X2); % PCA coefficientlarý ayný zamanda eigenvektörlerin büyükten küçüðe sýralý hali
Y2=coeff2'*(X2 - repmat(mx2,N2,1))'; % burada repmat mx ile ortalama deðer x ten çýkarýlarak resimdeki DC komponent yok ediliyor.
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
coeff2=pca(qqq); % PCA coefficientlarý ayný zamanda eigenvektörlerin büyükten küçüðe sýralý hali
Y2=coeff2'*(qqq - repmat(mx2,N2,1))'; % burada repmat mx ile ortalama deðer x ten çýkarýlarak resimdeki DC komponent yok ediliyor.
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