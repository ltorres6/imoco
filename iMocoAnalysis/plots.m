figure;
boxplot(T.tenengrad, T.reconType)
title('Tenengrad')

figure;
boxplot(T.tenengradX, T.reconType)
title('TenengradX')

figure;
boxplot(T.tenengradY, T.reconType)
title('TenengradY')

figure;
boxplot(T.tenengradZ, T.reconType)
title('TenengradZ')

figure;
boxplot(T.normalized_variance, T.reconType)
title('Normalized Variance')

figure;
boxplot(T.reduced_energy_ratio, T.reconType)
title('Reduced Energy Ratio')

figure;
boxplot(T.contrast_to_noise_aorta, T.reconType)
title('CNR Aorta')

figure;
boxplot(T.contrast_to_noise_lungs, T.reconType)
title('CNR Lung Parenchyma')

figure;
boxplot(T.contrast_to_noise_liver, T.reconType)
title('CNR Liver')

figure;
boxplot(T.apparent_snr_airway, T.reconType)
title('aSNR Airway')

figure;
boxplot(T.apparent_snr_lungs, T.reconType)
title('aSNR Lung Parenchyma')

figure;
boxplot(T.apparent_snr_liver, T.reconType)
title('aSNR Liver')

figure;
boxplot(T.apparent_snr_aorta, T.reconType)
title('aSNR Aorta')
