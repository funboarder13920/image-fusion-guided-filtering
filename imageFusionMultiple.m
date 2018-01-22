%% Load images
Iinit = im2double(imread('sourceimages\test4\Lenna.png'));
I1 = im2double(imread('sourceimages\test4\Lenna1.png'));
I2 = im2double(imread('sourceimages\test4\Lenna2.png'));
I = [];
I(1,:,:,:) = I1;
I(2,:,:,:) = I2;
figure(1);
imshow(squeeze(I(1,:,:,:)));

figure(2);
imshow(squeeze(I(2,:,:,:)));


%% Choice of parameters
r1 = 45;
eps1 = 0.3;
r2 = 7;
eps2 = 10^-6;

%% step A : two-scale image decomposition
% B1 and B2: blured images
% D1 and D2: detailed images

average_filter = 1/(31*31)*ones(31,31);

B = zeros(size(I));
D = zeros(size(I));
for i=1:size(I,1)
    B(i,:,:,:) = convn(squeeze(I(i,:,:,:)), average_filter, 'same');
    D(i,:,:,:) = squeeze(I(i,:,:,:) - B(i,:,:,:));
end

%% step B weight map construction

Ig = zeros(size(I(:,:,:,1)));
laplacian_filter = [[0 -1 0]; [-1 4 -1]; [0 -1 0]];
H = zeros(size(Ig));
S = zeros(size(Ig));
for i=1:size(I,1)
    %gray conversion
    Ig(i,:,:) = rgb2gray(squeeze(I(i,:,:,:)));
    % laplacian filtering
    H(i,:,:) = convn( squeeze(Ig(i,:,:)), laplacian_filter, 'same');
    % gaussian filtering
    S(i,:,:) = imgaussfilt(abs(squeeze(H(i,:,:))), 'FilterSize', 11);
end




% maybe reshape P1 and P2
P = zeros(size(S));
for i=1:size(I,1)
    P(i,:,:) = (S(i,:,:) == max(S));
end

% r1, eps1, r2 and eps2 are not related to the index of I1, P1, I2, P2, etc
Wb = zeros(size(P));
Wd = zeros(size(P));
for i=1:size(I,1)
    Wb(i,:,:) =  guidedfilter(squeeze(Ig(i,:,:)), squeeze(P(i,:,:)), r1, eps1);
    Wd(i,:,:) = guidedfilter(squeeze(Ig(i,:,:)), squeeze(P(i,:,:)), r2, eps2);
end

% normalizing weights
Sumb = sum(Wb,1);
Sumd = sum(Wd,1);
Wb = Wb./Sumb;
Wd = Wd./Sumd;


%% step C: two-scale image reconstruction

Bb = zeros(size(squeeze(I(1,:,:,:))));
Db = zeros(size(squeeze(I(1,:,:,:))));
for i=1:size(I,1)
    Bb = Bb + repmat(squeeze(Wb(i,:,:)), [1 1 3]) .* squeeze(B(i,:,:,:)); 
    Db = Db + repmat(squeeze(Wd(i,:,:)), [1 1 3]) .* squeeze(D(i,:,:,:)); 
end

F = Bb + Db;
figure(3);
imshow(F);

%figure(4);
%imshow([S1,P1,Wb1,Wd1],[0,1]);

%figure(5);
%imshow([S2,P2,Wb2,Wd2],[0,1]);

figure(6);
Idiff = F-Iinit;
imshow(Idiff-min(Idiff(:))/(max(Idiff(:))-min(Idiff(:))));