% Loop through subjects
clear all;
close all;
n = 3;
for k = [5,13,15,16,20,21,23]%,42]
    disp(['Subject: ', num2str(k)])
    basePath = ['/UserData/FainLab/testing/recon_comparisons/',num2str(k)];
    n = n+1;
    %% Load noGating
    I0 = load_nii(fullfile(basePath, ['noGate',num2str(k),'.nii.gz'])); I0 = I0.img;
    I0 = unityNormalization(I0);
    %% Load HardGating
    I1 = load_nii(fullfile(basePath, ['hardGate',num2str(k),'.nii.gz'])); I1 = I1.img;
    I1 = unityNormalization(I1);
    %% Load SoftGating
    I2 = load_nii(fullfile(basePath, ['softGate', num2str(k),'.nii.gz'])); I2 = I2.img;
    I2 = unityNormalization(I2);
    %% Load MoCo
    I3 = load_nii(fullfile(basePath, ['MoCo',num2str(k),'.nii.gz'])); I3 = I3.img;
    I3 = unityNormalization(I3);
    
    %% Load iMoCo
    I4 = load_nii(fullfile(basePath, ['iMoCo',num2str(k),'.nii.gz'])); I4 = I4.img;
    I4 = unityNormalization(I4);
    %     I3 = flip(flip(flip(I3,3),2),1); %[2, 3, 1])
    %     arrShow(I3); arrShow(I2)
    rois = load_nii(fullfile(basePath,'rois.nii.gz')); rois = rois.img;
    % liver = 1, lung = 2, airway = 3, heart = 4, background = 5
    
    
    %% Means
    %Calculate Liver Means
    L0(n) = mean(I0(rois == 1));
    L1(n) = mean(I1(rois == 1));
    L2(n) = mean(I2(rois == 1));
    L3(n) = mean(I3(rois == 1));
    L4(n) = mean(I4(rois == 1));
    
    %Calculate Lung P. Means
    Lu0(n) = mean(I0(rois == 2));
    Lu1(n) = mean(I1(rois == 2));
    Lu2(n) = mean(I2(rois == 2));
    Lu3(n) = mean(I3(rois == 2));
    Lu4(n) = mean(I4(rois == 2));
    
    %Calculate Airway Means
    A0(n) = mean(I0(rois == 3));
    A1(n) = mean(I1(rois == 3));
    A2(n) = mean(I2(rois == 3));
    A3(n) = mean(I3(rois == 3));
    A4(n) = mean(I4(rois == 3));
    
    %Calculate Aorta Means
    H0(n) = mean(I0(rois == 4));
    H1(n) = mean(I1(rois == 4));
    H2(n) = mean(I2(rois == 4));
    H3(n) = mean(I3(rois == 4));
    H4(n) = mean(fileTypes = [
    "PreContrastNoGate",
    "PreContrastHardGate",
    "PreContrastSoftGate",
    "PreContrastMoCo",
    "PreContrastIterativeMoCo",
    "PreContrastMoCoInsp",
    "PreContrastIterativeMoCoInsp",

]
fileNames = [
    "noGate_gw.nii.gz",
    "hardGate_gw.nii.gz",
    "softGate_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
    "MoCo_gw.nii.gz",
    "iMoCo_gw.nii.gz",
]I4(rois == 4));
    
    % Calculate Background StdDev
    B0(n) = std(I0(rois == 5));
    B1(n) = std(I1(rois == 5));
    B2(n) = std(I2(rois == 5));
    B3(n) = std(I3(rois == 5));
    B4(n) = std(I4(rois == 5));
    
    % Calculate Background Mean
    BM0(n) = mean(I0(rois == 5));
    BM1(n) = mean(I1(rois == 5));
    BM2(n) = mean(I2(rois == 5));
    BM3(n) = mean(I3(rois == 5));
    BM4(n) = mean(I4(rois == 5));
    
    %% Calculate Apparent SNR
    % Liver
    aSNR_L0(n) = L0(n)/B0(n);
    aSNR_L1(n) = L1(n)/B1(n);
    aSNR_L2(n) = L2(n)/B2(n);
    aSNR_L3(n) = L3(n)/B3(n);
    aSNR_L4(n) = L4(n)/B4(n);
    
    % Lungs
    aSNR_Lu0(n) = Lu0(n)/B0(n);
    aSNR_Lu1(n) = Lu1(n)/B1(n);
    aSNR_Lu2(n) = Lu2(n)/B2(n);
    aSNR_Lu3(n) = Lu3(n)/B3(n);
    aSNR_Lu4(n) = Lu4(n)/B4(n);
    
    % Airway
    aSNR_A0(n) = A0(n)/B0(n);
    aSNR_A1(n) = A1(n)/B1(n);
    aSNR_A2(n) = A2(n)/B2(n);
    aSNR_A3(n) = A3(n)/B3(n);
    aSNR_A4(n) = A4(n)/B4(n);
    
    % Aorta
    aSNR_H0(n) = H0(n)/B0(n);
    aSNR_H1(n) = H1(n)/B1(n);
    aSNR_H2(n) = H2(n)/B2(n);
    aSNR_H3(n) = H3(n)/B3(n);
    aSNR_H4(n) = H4(n)/B4(n);
    
    %% Calculate CNR
    % Airway
    CNR_A0(n) = (A0(n)-A0(n))/B0(n);
    CNR_A1(n) = (A1(n)-A1(n))/B1(n);
    CNR_A2(n) = (A2(n)-A2(n))/B2(n);
    CNR_A3(n) = (A3(n)-A3(n))/B3(n);
    CNR_A4(n) = (A4(n)-A4(n))/B4(n);
    
    % Liver
    CNR_L0(n) = (L0(n)-A0(n))/B0(n);
    CNR_L1(n) = (L1(n)-A1(n))/B1(n);
    CNR_L2(n) = (L2(n)-A2(n))/B2(n);
    CNR_L3(n) = (L3(n)-A3(n))/B3(n);
    CNR_L4(n) = (L4(n)-A4(n))/B4(n);
    
    % Lungs
    CNR_Lu0(n) = (Lu0(n)-A0(n))/B0(n);
    CNR_Lu1(n) = (Lu1(n)-A1(n))/B1(n);
    CNR_Lu2(n) = (Lu2(n)-A2(n))/B2(n);
    CNR_Lu3(n) = (Lu3(n)-A3(n))/B3(n);
    CNR_Lu4(n) = (Lu4(n)-A4(n))/B4(n);
    
    % Aorta
    CNR_H0(n) = (H0(n)-A0(n))/B0(n);
    CNR_H1(n) = (H1(n)-A1(n))/B1(n);
    CNR_H2(n) = (H2(n)-A2(n))/B2(n);
    CNR_H3(n) = (H3(n)-A3(n))/B3(n);
    CNR_H4(n) = (H4(n)-A4(n))/B4(n);
    
    
    %% Calculate Focus Measures
    %Normalize to Liver signal intensity
    I0 = I0./L0(n);
    I1 = I1./L1(n);
    I2 = I2./L2(n);
    I3 = I3./L3(n);
    I4 = I4./L4(n);
    
    thresh = -999999999;
    [Gx0, Gy0, Gz0] = imgradientxyz(I0, 'sobel');
%     Gx0 = abs(Gx0);
%     Gy0 = abs(Gy0);
%     Gz0 = abs(Gz0);
    G0 = Gx0.^2 + Gy0.^2 +Gz0.^2;
    
    [Gx1, Gy1, Gz1] = imgradientxyz(I1, 'sobel');
%     Gx1 = abs(Gx1);
%     Gy1 = abs(Gy1);
%     Gz1 = abs(Gz1);
    G1 = Gx1.^2 + Gy1.^2 +Gz1.^2;

    [Gx2, Gy2, Gz2] = imgradientxyz(I2, 'sobel');
%     Gx2 = abs(Gx2);
%     Gy2 = abs(Gy2);
%     Gz2 = abs(Gz2);
    G2 = Gx2.^2 + Gy2.^2 +Gz2.^2;

    [Gx3, Gy3, Gz3] = imgradientxyz(I3, 'sobel');
%     Gx3 = abs(Gx3);
%     Gy3 = abs(Gy3);
%     Gz3 = abs(Gz3);
    G3 = Gx3.^2 + Gy3.^2 +Gz3.^2;

    [Gx4, Gy4, Gz4] = imgradientxyz(I4, 'sobel');
%     Gx4 = abs(Gx4);
%     Gy4 = abs(Gy4);
%     Gz4 = abs(Gz4);
    G4 = Gx4.^2 + Gy4.^2 +Gz4.^2;
    
    FMT0x(n) = mean(Gx0(Gx0 > thresh));
    FMT1x(n) = mean(Gx1(Gx1 > thresh));
    FMT2x(n) = mean(Gx2(Gx2 > thresh));
    FMT3x(n) = mean(Gx3(Gx3 > thresh));
    FMT4x(n) = mean(Gx4(Gx4 > thresh));
    
    FMT0y(n) = mean(Gy0(Gy0 > thresh));
    FMT1y(n) = mean(Gy1(Gy1 > thresh));
    FMT2y(n) = mean(Gy2(Gy2 > thresh));
    FMT3y(n) = mean(Gy3(Gy3 > thresh));
    FMT4y(n) = mean(Gy4(Gy4 > thresh));
    
    FMT0z(n) = mean(Gz0(Gz0 > thresh));
    FMT1z(n) = mean(Gz1(Gz1 > thresh));
    FMT2z(n) = mean(Gz2(Gz2 > thresh));
    FMT3z(n) = mean(Gz3(Gz3 > thresh));
    FMT4z(n) = mean(Gz4(Gz4 > thresh));
    
    FMT0(n) = mean(G0(G0 > thresh));
    FMT1(n) = mean(G1(G1 > thresh));
    FMT2(n) = mean(G2(G2 > thresh));
    FMT3(n) = mean(G3(G3 > thresh));
    FMT4(n) = mean(G4(G4 > thresh));

    wsize=16;
    FMD0(n) = reRatio(I0,wsize);
    FMD1(n) = reRatio(I1,wsize);
    FMD2(n) = reRatio(I2,wsize);
    FMD3(n) = reRatio(I3,wsize);
    FMD4(n) = reRatio(I4,wsize);
    
    FMV0(n) = nVariance(I0);
    FMV1(n) = nVariance(I1);
    FMV2(n) = nVariance(I2);
    FMV3(n) = nVariance(I3);
    FMV4(n) = nVariance(I4);
end
g0 = repmat({'noGate'}, size(aSNR_A0, 2),1);
g1 = repmat({'hardGate'}, size(aSNR_A1, 2),1);
g2 = repmat({'softGate'}, size(aSNR_A2, 2),1);
g3 = repmat({'MoCo'}, size(aSNR_A3, 2),1);
g4 = repmat({'iMoCo'}, size(aSNR_A4, 2),1);
g = [g0;g1;g2;g3;g4];
xlabels = {'noGate', 'HardGate', 'SoftGate','MoCo', 'iMoCo'};

%% SNR BoxPLots
% Airway
aSNR_A = [aSNR_A0', aSNR_A1', aSNR_A2', aSNR_A3', aSNR_A4'];
[~, ~, stats] = anova1(aSNR_A(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(aSNR_A);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Airway aSNR');
xticklabels(xlabels);

% Liver
aSNR_L = [aSNR_L0', aSNR_L1', aSNR_L2', aSNR_L3', aSNR_L4'];
[~, ~, stats] = anova1(aSNR_L(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(aSNR_L);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Liver aSNR');
xticklabels(xlabels);

% Lung Parenchyma
aSNR_Lu = [aSNR_Lu0', aSNR_Lu1', aSNR_Lu2', aSNR_Lu3', aSNR_Lu4'];
[~, ~, stats] = anova1(aSNR_Lu(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(aSNR_Lu);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Lung Parenchyma aSNR');
xticklabels(xlabels);

%Aorta
aSNR_H = [aSNR_H0', aSNR_H1', aSNR_H2', aSNR_H3', aSNR_H4'];
[~, ~, stats] = anova1(aSNR_H(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(aSNR_H);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Aorta aSNR');
xticklabels(xlabels);

%% CNR BoxPLots
% Airway
CNR_A = [CNR_A0', CNR_A1', CNR_A2', CNR_A3', CNR_A4'];
[~, ~, stats] = anova1(CNR_A(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(CNR_A);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Airway CNR');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'AirwayCNR.tiff', 'tiff');

% Liver
CNR_L = [CNR_L0', CNR_L1', CNR_L2', CNR_L3', CNR_L4'];
[~, ~, stats] = anova1(CNR_L(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(CNR_L);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Liver CNR');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'LiverCNR.tiff', 'tiff');

% Lung Parenchyma
CNR_Lu = [CNR_Lu0', CNR_Lu1', CNR_Lu2', CNR_Lu3', CNR_Lu4'];
[~, ~, stats] = anova1(CNR_Lu(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(CNR_Lu);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Lung Parenchyma CNR');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'LungParenchymaCNR.tiff', 'tiff');

% Aorta
CNR_H = [CNR_H0', CNR_H1', CNR_H2', CNR_H3', CNR_H4'];
[~, ~, stats] = anova1(CNR_H(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(CNR_H);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Aorta CNR');
xticklabels(xlabels);

% figurePath = fullfile(saveDir,[qcCohort,'_',metric,'_adjusted.tiff']);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'AortaCNR.tiff', 'tiff');
%% Focus Measure
% Tenengrad
FMT = [FMT0',FMT1',FMT2',FMT3',FMT4'];
[~, ~, stats] = anova1(FMT(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMT);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Tenengrad Focus Measure');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'Tenengrad.tiff', 'tiff');

FMTx = [FMT0x',FMT1x',FMT2x',FMT3x',FMT4x'];
FMTy = [FMT0y',FMT1y',FMT2y',FMT3y',FMT4y'];
FMTz = [FMT0z',FMT1z',FMT2z',FMT3z',FMT4z'];

[~, ~, stats] = anova1(FMTx(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMTx);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Tenengrad Focus Measure X');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'TenengradX.tiff', 'tiff');

[~, ~, stats] = anova1(FMTy(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMTy);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Tenengrad Focus Measure Y');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'TenengradY.tiff', 'tiff');

[~, ~, stats] = anova1(FMTz(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMTz);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Tenengrad Focus Measure Z');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'TenengradZ.tiff', 'tiff');


% Reduced Energy Ratio (DCT)
FMD = [FMD0',FMD1',FMD2',FMD3',FMD4'];
[~, ~, stats] = anova1(FMD(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMD);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Reduced Energy Ratio Focus Measure');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'ReducedEnergy.tiff', 'tiff');

% Normalized Variance
FMV = [FMV0',FMV1',FMV2',FMV3',FMV4'];
[~, ~, stats] = anova1(FMV(:),g, 'off');
[c,~,~,~] = multcompare(stats,'Display','off');
figure;
notBoxPlot(FMV);
ind = find(c(:, 6) < 0.05);
groups = {};
if ~isempty(ind)
    for l = 1:numel(ind)
        groups = [groups, {[c(ind(l),1), c(ind(l), 2)]}];
    end
    sigstar(groups,c(ind,6), 'sort');
end
ylabel('Normalized Variance Focus Measure');
xticklabels(xlabels);
pause(1)
img = getframe(gcf);
imwrite(img.cdata, 'NormalizedVariance.tiff', 'tiff');

% % Normalized Variance Measure (Statistical)
% figure;
% notBoxPlot([FMV0', FMV1',FMV2',FMV3'])
% xticklabels({'noGate', 'HardGate', 'SoftGate','iMoCo'})
% ylabel('Normalized Variance Focus Measure');


%     binmax = 2;
%     figure;
%     histogram(G0(:), 'BinLimits',[0,binmax]); hold on;
%     histogram(G1(:), 'BinLimits',[0,binmax]);
%     histogram(G2(:), 'BinLimits',[0,binmax]);
%     histogram(G3(:), 'BinLimits',[0,binmax]);
%     hold off;
%     legend('noGate', 'HardGate', 'softGate', 'iMoCo');
%     drawnow;