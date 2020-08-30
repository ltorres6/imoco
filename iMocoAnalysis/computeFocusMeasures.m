function [rer, nVariance, tenengrad,tenengradX,tenengradY,tenengradZ, wave_fm] =  computeFocusMeasures(img, wsize, thresh, img_std)
% This function computes the focus measures for a 3D image.
% Images should be normalized to a reference tissue for comparison across
% subjects.

% Reduced Energy Ratio Focus Measure
I_dct = mirt_dctn(img);
rer = 0;
for k = 1:wsize
    for l = 1:wsize
        for m = 1:wsize
            rer = rer + I_dct(k,l,m).^2;
        end
    end
end
rer = rer - I_dct(1,1,1).^2;
rer = rer/(I_dct(1,1,1).^2);

% Normalized Variance Focus Measure
avg = mean(img(:));
nVariance = (img(:)-avg).^2;
nVariance = sum(nVariance);
nVariance = nVariance/numel(img);
nVariance = nVariance/avg;


% Tenengrad Focus Measure
% thresh = -999999;
[Gx, Gy, Gz] = imgradientxyz(img, 'sobel');
tenengrad = Gx.^2 + Gy.^2 +Gz.^2;
tenengrad = mean(tenengrad(tenengrad > thresh));
tenengradX = mean(Gx(Gx > thresh));
tenengradY = mean(Gy(Gy > thresh));
tenengradZ = mean(Gz(Gz > thresh));

% Wavelet Decomposition
n = 2; % Decomposition Level
w = 'db4'; % Wavelet Type
waveDecomp = wavedec3(img, n, w);

% Extract Coeff
cL = cell(1,n); % Low pass
cH = cell(1,n); % High pass
for k = 1:n
    cL{k} = waverec3(waveDecomp,'ca',k);   % Approximations (low-pass components)
   
    cH{k} = waverec3(waveDecomp,'cd',k);   % Details (high-pass components)
end
level=2; % Second Level More robust to noise

H = cH{level}(:);
L = cL{level}(:);

noise_thresh = img_std*sqrt(2*log(numel(img))/numel(img));
H(H < noise_thresh) = 0;

H = sum(abs(H).^2);
L = sum(abs(L).^2);

wave_fm = H/L;

end


