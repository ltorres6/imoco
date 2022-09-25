% Register multiphase UTE mri images

mrimg = load_nii('/home/ltorres/data/recon/ipf/103-002/mri/20151222/pre_contrast/MotionResolved_test/MotionResolvedLowRes.nii.gz');
mrimg = abs(mrimg.img);
mrimg = squeeze(mrimg)./max(abs(mrimg(:)));
mrimg = permute(mrimg, [2,3,4,1]);
m_ph = size(mrimg, 4);
fixed = mrimg(:,:,:,1);
warped = zeros(size(mrimg));
for i = 3:3
    [reg_fieldt, warped(:,:,:,i)] = imregdemons(mrimg(:,:,:,i),fixed,'PyramidLevels',4,'AccumulatedFieldSmoothing',2,'DisplayWaitbar',true);
end