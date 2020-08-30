function fm = nVariance(I)
I = unityNormalization(I);
avg = mean(I(:));
fm = I(:)-avg;
fm = sum(fm);
fm = fm/avg;
end