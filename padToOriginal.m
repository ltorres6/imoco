function out = padToOriginal(I)
    % This script pads iMoCo recons to 512^3, then crops to 256^3 for spatial
    % alignment with other recon outputs (hardgating, softgating).
    inSize = size(I);
    padSize = round((512 - inSize)/2);
    cropSize = 129:129+255;
    out = padarray(I, padSize);
    out = out(cropSize, cropSize, cropSize);
    end