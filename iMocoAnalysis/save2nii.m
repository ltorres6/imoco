% Loop through subjects
clear all;
close all;
codeDir=pwd; % Start in code directory
for k = [13,15,16,20,23]
    disp(['Processing subject: ', num2str(k)])
    basePath = ['/UserData/FainLab/testing/recon_comparisons/',num2str(k)];
    %% Reconstruct Using HardGating
    cd(basePath)
%     %% Reconstruct Using noGating
%     call = '!pcvipr_recon_binary -f P*';
%     evalc(call);
%     I = readBinaryUte('X_000_000.dat');
%     I = flip(flip(flip(permute(I, [2, 3, 1]),3),2),1);
%     save_nii(make_nii(I), ['noGate',num2str(k),'.nii.gz'])
%     delete('X_000_000.dat')
% %         sudo -E env "PATH=$PATH"
%     call = '!pcvipr_recon_binary -f P* -resp_gate thresh -correct_resp_drift -resp_gate_signal bellows -resp_gate_efficiency 0.5 -export_kdata';
%     evalc(call);
%     I = readBinaryUte('X_000_000.dat');
%     I = flip(flip(flip(permute(I, [2, 3, 1]),3),2),1);
%     save_nii(make_nii(I), ['hardGate',num2str(k),'.nii.gz'])
%     delete('X_000_000.dat')
% % %         arrShow(I)
%     %% Reconstruct Using SoftGating
%     call = '!pcvipr_recon_binary -f P* -resp_gate weight -correct_resp_drift -resp_gate_signal bellows -resp_gate_efficiency 0.10 -resp_gate_weight 5';
%     evalc(call);
%     I = readBinaryUte('X_000_000.dat');
%     I = flip(flip(flip(permute(I, [2, 3, 1]),3),2),1);
%     save_nii(make_nii(I), ['softGate',num2str(k),'.nii.gz'])
%     delete('X_000_000.dat')

    %     arrShow(I)
    %% Reconstruct Using iMoCo
%     cd(codeDir);
%     Isane = load_nii(['softGate',num2str(k), '.nii.gz']); Isane = Isane.img;

    I = load_nii(['iMoCo',num2str(k), '.nii.gz']); I = I.img;
%     I1 = pad_imoco(abs(I_imoco));
    I1 = flip(flip(flip(I,3),2),1);
    save_nii(make_nii(I1), ['iMoCo',num2str(k),'.nii.gz'])

%     I2 = pad_imoco(abs(I_moco));
%     I2 = flip(flip(flip(I2,3),2),1);
% 
%     save_nii(make_nii(I1), ['iMoCo',num2str(k),'.nii.gz'])
%     save_nii(make_nii(I2), ['MoCo',num2str(k),'.nii.gz'])
%     I = load_nii(['iMoCo', num2str(k),'.nii.gz']); I = I.img;
%     load('MRI_Raw_imoco_pd6.mat');
%     I = flip(flip(flip(I,3),2),1);
%     save_nii(make_nii(I), ['iMoCo',num2str(k),'.nii.gz'])
    
%     system(['cp *.nii.gz /UserData/FainLab/testing/recon_comparisons/', num2str(k)])
    
end