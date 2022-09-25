% subject = '103-005';
% visit = '20160114';
% cSlice = 80;
% aSlice = 97;
% sSlice = 89;

% subject = '103-016';
% visit = '20161121';
% cSlice = 120;
% aSlice = 123;
% sSlice = 161;

subject = '103-042';
visit = '20190221';
cSlice = 100;
aSlice = 112;
sSlice = 90;
scale = 0.2
basePath = fullfile('~/data/ipf/ltorres', subject, 'mri', visit);
figurePath = fullfile(basePath, 'figures');
fileTypes = {'PreContrastNoGate', 'PreContrastHardGate', 'PreContrastSoftGate', 'PreContrastMoCo','PreContrastMoCoInsp','PreContrastIterativeMoCo', 'PreContrastIterativeMoCoInsp'};
fileNames = {'noGate_gw.nii.gz', 'hardGate_gw.nii.gz', 'softGate_gw.nii.gz', 'MoCo_gw.nii.gz', 'MoCo_gw.nii.gz', 'iMoCo_gw.nii.gz', 'iMoCo_gw.nii.gz'};
disp(['Working on subject:', subject])
% img = niftiread(fullfile(basePath,fileTypes{7}, fileNames{7}));
% arrShow(img)
for ii = 1:numel(fileTypes)
    outCoronalName = fullfile(figurePath, [fileTypes{ii}, num2str(cSlice), 'Coronal.png']);
    outAxialName = fullfile(figurePath, [fileTypes{ii}, num2str(aSlice), 'Axial.png']);
    outSaggitalName = fullfile(figurePath, [fileTypes{ii}, num2str(sSlice), 'Saggital.png']);
    img = niftiread(fullfile(basePath,fileTypes{ii}, fileNames{ii}));
    % Axial
    imshow(fliplr(rot90(img(:,:,aSlice),1)), [0, scale*max(img(:))],'Border','tight')
    frame = getframe(gcf);
    imwrite(frame.cdata, outAxialName);
    pause(0.5)
    close
    
    %Coronal
    imshow(fliplr(rot90(squeeze(img(:,cSlice,:)),1)), [0, scale*max(img(:))],'Border','tight')
    frame = getframe(gcf);
    imwrite(frame.cdata, outCoronalName);
    pause(0.5)
    close
    
    %Saggital
    imshow(fliplr(rot90(squeeze(img(sSlice,:,:)),1)), [0, scale*max(img(:))],'Border','tight')
    frame = getframe(gcf);
    imwrite(frame.cdata, outSaggitalName);
    pause(0.5)
    close
end
