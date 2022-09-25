% Loop through subjects and compute CNR and SNR.
clear all;
close all;

% Add tools and path
addpath('/data/users/ltorres/projects/xdgrasp')
addpath(genpath('/data/users/ltorres/projects/tools'));
codeDir = '/export/home/ltorres/projects/imoco'; % Start in code directory
load(fullfile(codeDir, 'subject_ids.mat'))
setenv('TMP', '/export/home/ltorres/tmp/')
fileTypes = {'PreContrastNoGate', 'PreContrastHardGate', 'PreContrastSoftGate', 'PreContrastMoCo','PreContrastIterativeMoCo'};
fileNames = {'noGate_gw.nii.gz', 'hardGate_gw.nii.gz', 'softGate_gw.nii.gz', 'MoCo_gw.nii.gz', 'iMoCo_gw.nii.gz'};
% Loop through each subject
n = 0;
for l = 1:numel(subjects.subject_id)
    disp(subjects.subject_id(l))
    
    % Select IPF subjects
    if strcmp(char(subjects.cohort_id(l)), 'IPF') %IPF or negate for healthy
        subject = char(subjects.subject_id(l));
        visit = char(subjects.visit_id(l));
        basePath = fullfile('/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf', subject, 'mri', visit);
        disp(['Working on subject:', subject])
        % Check if iMoCo Exists
        if ~exist(fullfile(basePath, 'PreContrastIterativeMoCo', 'regionsOfInterest.nii.gz'), 'file')
            disp(['regionsOfInterest.nii.gz does not exist for subject ', subject, ', visit ', visit])
            continue
        end
        % Load ROIs for subject at this visit.
        rois = load_nii(fullfile(basePath, 'PreContrastIterativeMoCo', 'regionsOfInterest.nii.gz')); rois = flip(flip(rois.img,2),1);
        
        % Loop through each image
        for m = 1:numel(fileTypes)
            imgPath = fullfile(basePath, fileTypes{m}, fileNames{m});
            
            % Load image
            img = load_nii(imgPath); img = img.img;
            % img = (img- min(img(:)))/(max(img(:)) - min(img(:)));
            % liver = 1, lung = 2, airway = 3, heart = 4, background = 5
            
            %% Calculate Means
            % Calculate Liver mean
            liver_mean = mean(img(rois == 1));
            
            %Calculate Lung Parenchyma Mean
            lung_mean = mean(img(rois == 2));
            
            %Calculate Airway Means
            airway_mean = mean(img(rois == 3));
            
            %Calculate Aorta Means
            aorta_mean = mean(img(rois == 4));
            
            % Calculate Background StdDev
            background_mean = mean(img(rois == 5));
            background_std = std(img(rois == 5));
            
            %% Calculate Apparent SNR
            % Liver
            apparent_snr_liver = liver_mean/background_std;
            
            % Lungs
            apparent_snr_lungs = lung_mean/background_std;
            
            % Airway
            apparent_snr_airway = airway_mean/background_std;
            
            % airway
            apparent_snr_aorta = aorta_mean/background_std;
            
            
            %% Calculate CNR with respect to airway
            % Liver
            contrast_to_noise_liver = apparent_snr_liver - apparent_snr_airway;
            
            % Lungs
            contrast_to_noise_lungs = apparent_snr_lungs - apparent_snr_airway;
            
            % Aorta
            contrast_to_noise_aorta = apparent_snr_aorta - apparent_snr_airway;
            
            %% Compute Focus Measures
            % Normalize to liver tissue signal
            img=img./liver_mean;
            dct_wsize=100;
            tenengrad_thresh = 2*background_std;
            dwt_thresh= background_std;
            [reduced_energy_ratio, normalized_variance, tenengrad,tenengradX,tenengradY,tenengradZ, wave_fm] =  computeFocusMeasures(img, dct_wsize, tenengrad_thresh, dwt_thresh);
            
            %% Generate Table
            reconType = extractBefore(fileNames{m}, '.');
            % Arrange in cell array to export as table
            cells = {subject, visit, reconType, liver_mean, lung_mean, airway_mean, aorta_mean,...
                background_mean, background_std, apparent_snr_liver, apparent_snr_lungs,...
                apparent_snr_airway, apparent_snr_aorta, contrast_to_noise_liver, ...
                contrast_to_noise_lungs, contrast_to_noise_aorta, reduced_energy_ratio,...
                normalized_variance, tenengrad, tenengradX, tenengradY, tenengradZ, wave_fm};
            
            if n==0
                T = cell2table(cells);
                n = n+1;
            else
                T = [T; cell2table(cells)];
            end
        end
    end
end
varNames = {'subject','visit', 'reconType', 'liver_mean', 'lung_mean', 'airway_mean', 'aorta_mean',...
                'background_mean', 'background_std', 'apparent_snr_liver', 'apparent_snr_lungs',...
                'apparent_snr_airway', 'apparent_snr_aorta', 'contrast_to_noise_liver', ...
                'contrast_to_noise_lungs', 'contrast_to_noise_aorta', 'reduced_energy_ratio',...
                'normalized_variance', 'tenengrad', 'tenengradX', 'tenengradY', 'tenengradZ','wave_fm'};
T.Properties.VariableNames = varNames;

writetable(T, 'imocoMetrics.xlsx')