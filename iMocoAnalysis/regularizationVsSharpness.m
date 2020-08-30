% Plot regularization parameter vs metrics.
addpath(genpath('/data/users/ltorres/projects/tools'));
% 
% function a = mymean(v,n)
% % MYMEAN Local function that calculates mean of array.
% 
%     a = sum(v)/n;
% end


subject = '103-005';
visit = '20160114';
basePath = fullfile('/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf', subject, 'mri', visit, 'PreContrastIterativeMoCo');
rois = niftiread(fullfile(basePath, 'regionsOfInterest.nii.gz'));% rois = flip(flip(rois,2),1);
lambdas = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
for k = 1:length(lambdas)
    curImg=fullfile(basePath, ['iMoCo_', num2str(lambdas(k)),'.nii.gz'])
    img = niftiread(curImg);
    liver_mean = mean(img(rois == 1));
    background_std = std(img(rois == 5));
    background_std = -99999;
    img=img./liver_mean;
    dct_wsize=100;
    tenengrad_thresh = 2*background_std;
    dwt_thresh= background_std;
    [rer(k), nVariance(k), tenengrad(k),tenengradX(k),tenengradY(k),tenengradZ(k), wave_fm(k)] =  computeFocusMeasures(img, dct_wsize, tenengrad_thresh, dwt_thresh);
end

%%
subplot(7,1,1)
plot(lambdas, rer)
title('Reduced Energy Ratio')

subplot(7,1,2)
plot(lambdas, nVariance);
title('Normalized Variance')

subplot(7,1,3)
plot(lambdas, tenengrad);
title('Tenengrad')

subplot(7,1,4)
plot(lambdas, tenengradX);
title('TenengradX')

subplot(7,1,5)
plot(lambdas, tenengradY);
title('TenengradY')

subplot(7,1,6)
plot(lambdas, tenengradZ);
title('TenengradZ')

subplot(7,1,7)
plot(lambdas, wave_fm);
title('Wavelet')

