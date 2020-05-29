% Loop through subjects
% clear all;
% close all;
addpath('/data/users/ltorres/projects/imoco')
codeDir = '/export/home/ltorres/projects/imoco'; % Start in code directory
load(fullfile(codeDir, 'subject_ids.mat'))
n_threads = '12';
n_threads_moco = '12';
matThreads = 12;
setenv('OMP_NUM_THREADS', n_threads)
maxNumCompThreads(matThreads);
% setenv('TMP', '/export/home/ltorres/tmp/')

for l = 37:numel(subjects.subject_id)
    disp(subjects.subject_id(l))

    if strcmp(char(subjects.cohort_id(l)), 'IPF')%IPF or negate for healthy
        subject = char(subjects.subject_id(l));
        visit = char(subjects.visit_id(l));

        basePath = fullfile('/data/data_mrcv2/FAIN_GROUP/FainLab/recon/ipf', subject, 'mri', visit);
        fileTypes = {'PreContrastNoGate', 'PreContrastHardGate', 'PreContrastSoftGate', 'PreContrastMoCo','PreContrastMoCoInsp','PreContrastIterativeMoCo', 'PreContrastIterativeMoCoInsp'};
        fileNames = {'noGate.nii.gz', 'hardGate.nii.gz', 'softGate.nii.gz', 'MoCo.nii.gz', 'MoCo.nii.gz', 'iMoCo.nii.gz', 'iMoCo.nii.gz'};
        disp(['Working on subject:', subject])
        % Copy Data
        if exist(fullfile(basePath, 'PreContrastRetrospectiveUTE'), 'dir')

            for ii = 1:numel(fileTypes)
                outPath = fullfile(basePath, fileTypes{ii});

                if exist(fullfile(outPath, fileNames{ii}), 'file')
                    gradwarp(outPath, fileNames{ii}, fullfile(basePath, 'PreContrastRetrospectiveUTE'))
                else
                    disp(["File Missing for Subject: ", subject])
                end

            end

        end

    end

end
