% Loop through subjects
% clear all;
% close all;
addpath('/data/users/ltorres/projects/xdgrasp')
codeDir = '/export/home/ltorres/projects/imoco'; % Start in code directory
load(fullfile(codeDir, 'subject_ids.mat'))
n_threads = '12';
n_threads_moco = '12';
matThreads = 12;
setenv('OMP_NUM_THREADS', n_threads)
maxNumCompThreads(matThreads);
% setenv('TMP', '/export/home/ltorres/tmp/')

for l = 1:numel(subjects.subject_id)
    disp(subjects.subject_id(l))

    if strcmp(char(subjects.cohort_id(l)), 'IPF')%IPF or negate for healthy
        subject = char(subjects.subject_id(l));
        visit = char(subjects.visit_id(l));

        basePath = fullfile('/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf', subject, 'mri', visit);
        fileTypes = {'PreContrastNoGate', 'PreContrastHardGate', 'PreContrastSoftGate', 'PreContrastMoCo',...
            'PreContrastMoCoInsp','PreContrastIterativeMoCo', 'PreContrastIterativeMoCoInsp',...
            'PostContrastNoGate', 'PostContrastHardGate', 'PostContrastSoftGate', 'PostContrastMoCo',...
            'PostContrastMoCoInsp','PostContrastIterativeMoCo', 'PostContrastIterativeMoCoInsp'};
        fileNames = {'noGate.nii.gz', 'hardGate.nii.gz', 'softGate.nii.gz', 'MoCo.nii.gz', 'MoCo.nii.gz', 'iMoCo.nii.gz', 'iMoCo.nii.gz',...
            'noGate.nii.gz', 'hardGate.nii.gz', 'softGate.nii.gz', 'MoCo.nii.gz', 'MoCo.nii.gz', 'iMoCo.nii.gz', 'iMoCo.nii.gz'};
        fileNames_gw = {'noGate_gw.nii.gz', 'hardGate_gw.nii.gz', 'softGate_gw.nii.gz', 'MoCo_gw.nii.gz', 'MoCo_gw.nii.gz', 'iMoCo_gw.nii.gz', 'iMoCo_gw.nii.gz',...
            'noGate_gw.nii.gz', 'hardGate_gw.nii.gz', 'softGate_gw.nii.gz', 'MoCo_gw.nii.gz', 'MoCo_gw.nii.gz', 'iMoCo_gw.nii.gz', 'iMoCo_gw.nii.gz'};
        disp(['Working on subject:', subject])
        % Copy Data
        if exist(fullfile(basePath, 'PreContrastRetrospectiveUTE'), 'dir') || exist(fullfile(basePath, 'PostContrastRetrospectiveUTE'), 'dir')
            for ii = 1:numel(fileTypes)
                outPath = fullfile(basePath, fileTypes{ii});
                if strncmpi('pre', fileTypes{ii},3)
                    pcviprHeaderLocation='PreContrastRetrospectiveUTE';
                elseif strncmpi('pos', fileTypes{ii},3)
                    pcviprHeaderLocation='PostContrastRetrospectiveUTE';
                else
                    error('no match, do not know where to find pcviprheader.txt')
                end
                
                if exist(fullfile(outPath, fileNames{ii}), 'file')
                    disp(['Gradwarping ', outPath])
                    if ~exist(fullfile(outPath, fileNames_gw{ii}), 'file')
                        gradwarp(outPath, fileNames{ii}, fullfile(basePath, pcviprHeaderLocation))
                    end
                else
                    disp([fileNames{ii}, " Missing for Subject: ", subject, ", Visit: ", visit])
                end

            end
        else
            disp('PreContrastRetrospective nor PostContrastRetrospective seem to exist...')
        end

    end

end
PA4eturn Linop