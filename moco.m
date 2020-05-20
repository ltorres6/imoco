function Ix = moco(basePath, ref)

    disp('Reading Reconstructed Data. ...')
    mr_img = niftiread([basePath, 'MotionResolved.nii.gz']);
    mr_img = permute(mr_img, [3, 2, 1, 4]);
    IsizeL = size(mr_img);
    disp(IsizeL)
    nMotionStates = IsizeL(end);
    IsizeL = IsizeL(1:3);
    Isize = IsizeL;
    mask = ones(IsizeL);

    % estimate motion state
    mr_img = squeeze(mr_img) ./ max(abs(mr_img(:)));
    mag_b = abs(imgauss4d(mr_img, .5));
    % motion field interpolation
    mscale = Isize ./ IsizeL;
    reg_field = zeros([Isize(1:3), 3, nMotionStates]); % [rx, ry, rz, dims, motionstate]
    % reg_field update
    % reg_field2 = reg_field;
    Ix = zeros(IsizeL(1:3));
    disp('Registration Start. ...')
    if gpuDeviceCount >= 1
        gpu_exist = true;
        g = gpuDevice(1);
        disp("Using GPU")
    else
        gpu_exist = false;
    end
    a1 = tic;
    for i = 1:nMotionStates
        disp(['Registering Motion Phase ', num2str(i), '...'])

        if gpu_exist == false
            [reg_fieldt] = imregdemons(mag_b(:, :, :, i), mag_b(:, :, :, ref), 'PyramidLevels', 4, 'DisplayWaitbar', false); %Register each motion state
        else
            %----------------------- On GPU
            [reg_fieldGPU] = imregdemons(gpuArray(mag_b(:, :, :, i)), gpuArray(mag_b(:, :, :, ref)), 'PyramidLevels', 4, 'DisplayWaitbar', false); %Register each motion state
            reg_fieldt = gather(reg_fieldGPU);
            %-----------------
        end

        Ix = Ix + imwarp(mr_img(:, :, :, i), reg_fieldt); %Sum Motion States, will divide by number of states later for average.
        %     for k = 1:3
        %        reg_fieldt2(:,:,:,k) = imwarp(reg_fieldt(:,:,:,k),reg_fieldt);
        %     end
        for j = 1:3
            reg_field(:, :, :, j, i) = imresize3(reg_fieldt(:, :, :, j) .* mask * mscale(j), Isize);
            %         reg_field2(:,:,:,j,i) = imresize3(reg_fieldt2(:,:,:,j).*mask*mscale(j),Isize); %Accumulate registrations?
        end

    end

    b1 = toc(a1) / 60;
    disp(['Registration Done. Time:', num2str(b1), ' min.'])
    Ix = Ix ./ nMotionStates;

    if gpu_exist == true
        reset(g);
    end
    Ix = padToOriginal(Ix);
    Ix = flip(flip(flip(Ix,3),2),1);
end
