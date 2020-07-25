clear all;
close all;
for k = [13,15,16,20,23,42]
    basePath = ['/UserData/FainLab/testing/recon_comparisons/',num2str(k)];
    I3 = load_nii(fullfile(basePath, ['iMoCo',num2str(k),'.nii.gz'])); I3 = I3.img;
    I3 = unityNormalization(I3);
    %% look at image and select slice
    imshow3Dfull(I3, [0 0.25]);
    pause(1);
    
end