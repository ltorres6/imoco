function varargout = gradwarp(varargin)

    if nargin == 0
        [filename, pathname] = uigetfile('*.*');
    elseif nargin == 3
        pathname = varargin{1};
        filename = varargin{2};
        headerpath = varargin{3};
    else
        error("Incorrect Number of Arguments")
    end

    image_fullpath = fullfile(pathname, filename);
    header_fullpath = fullfile(headerpath, 'pcvipr_header.txt');

    if ~exist(header_fullpath, 'file')
        error('pcvipr_header.txt does not exist in the chosen folder');
    end

    info = read_header(header_fullpath);

    % The position vectors from PCVIPR
    U = info.U;
    S = info.O;
    % disp(S)
    %Put in center of bore
    ZS = U * [info.rcxres / 2; info.rcyres / 2; info.rczres / 2];
    S(end) = -ZS(end);

    %%Flip it for test. Looks the same so we need more to test this.
    % S = -S;
    % U = -U;
    % disp(S)
    [~, ~, fext] = fileparts(image_fullpath);

    if any(strcmp(fext, '.dat'))
        fid = fopen(image_fullpath);
        im_in = fread(fid, 'float');
        im_in = reshape(im_in, [info.rcxres info.rcyres info.rczres]);
        fclose(fid);
    elseif any(strcmp(fext, {'.nii', '.gz', '.nii.gz'}))
        %im_in = load_nii(image_fullpath);
        %im_in = flip(flip(flip(im_in.img, 1), 3), 2);
        im_in = niftiread(image_fullpath);
        im_in = flip(flip(flip(im_in, 1), 3), 2);
    else
        error('Unknown File Extension')
    end

    im_in = permute(im_in, [2 1 3]);

    corners.UpperLeft = (U * [0; 0; 0] + S)';
    corners.UpperRight = (U * [0; info.rcyres - 1; 0] + S)';
    corners.LowerLeft = (U * [info.rcxres - 1; 0; 0] + S)';
    corners.Type = 'SliceCorners';

    corners2.UpperLeft = (U * [0; 0; info.rczres - 1] + S)';
    corners2.UpperRight = (U * [0; info.rcyres - 1; info.rczres - 1] + S)';
    corners2.LowerLeft = (U * [info.rcxres - 1; 0; info.rczres - 1] + S)';
    corners2.Type = 'SliceCorners';

    im_out = GERecon('Gradwarp', im_in, [corners corners2], 'XRM');
    im_out = permute(im_out, [2 1 3]);

    if nargout == 1
        varargout{1} = im_out;
        return
    else
        disp("Writing...")
        % Make output filename with appendix '_gw'
        [filepath, filename, ~] = fileparts(image_fullpath);
        k = strfind(filename, '.');

        if ~isempty(k)
            filename = char(filename);
            k0 = k(1) - 1;
            outpath = fullfile(filepath, strcat(filename(1:k0), '_gw'));
        else
            outpath = fullfile(filepath, strcat(filename, '_gw'));
        end

        disp(["Outdir: ", outpath])

        if any(strcmp(fext, '.dat'))
            fid = fopen(strcat(outpath, '.dat'), 'w');
            fwrite(fid, im_out, 'float');
            fclose(fid);
        elseif any(strcmp(fext, {'.nii', '.gz', '.nii.gz'}))
            im_out = flip(flip(flip(im_out, 2), 3), 1);
            niftiwrite(im_out, outpath, 'Compressed', true)
        else
            error('Unknown File Extension')
        end

    end

    %beep

    function info = read_header(name)

        [parameter value] = textread(name, '%s %s');
        status = 1;

        %%%EXAM
        idx = find(strcmp('exam', parameter));
        info.exam = str2num(value{idx});

        %%%FOVS
        idx = find(strcmp('fovx', parameter));
        info.xfov = str2num(value{idx});

        idx = find(strcmp('fovy', parameter));
        info.yfov = str2num(value{idx});

        idx = find(strcmp('fovz', parameter));
        info.zfov = str2num(value{idx});

        %%%Matrix
        idx = find(strcmp('matrixx', parameter));
        info.rcxres = str2num(value{idx});

        idx = find(strcmp('matrixy', parameter));
        info.rcyres = str2num(value{idx});

        idx = find(strcmp('matrixz', parameter));
        info.rczres = str2num(value{idx});

        %%%Time Stuff
        idx = find(strcmp('frames', parameter));
        info.frames = str2num(value{idx});

        if (info.frames == -1)
            info.frames = 3;
        end

        idx = find(strcmp('timeres', parameter));
        info.tres = str2num(value{idx}) / 1000;

        % idx = find(strcmp('VENC',parameter));
        % info.VENC = str2num( value{idx} );

        idx = find(strcmp('version', parameter));
        info.version = str2num(value{idx});

        if info.version > 1
            idx = find(strcmp('ix', parameter));
            ix = str2num(value{idx});
            idx = find(strcmp('iy', parameter));
            iy = str2num(value{idx});
            idx = find(strcmp('iz', parameter));
            iz = str2num(value{idx});

            idx = find(strcmp('jx', parameter));
            jx = str2num(value{idx});
            idx = find(strcmp('jy', parameter));
            jy = str2num(value{idx});
            idx = find(strcmp('jz', parameter));
            jz = str2num(value{idx});

            idx = find(strcmp('kx', parameter));
            kx = str2num(value{idx});
            idx = find(strcmp('ky', parameter));
            ky = str2num(value{idx});
            idx = find(strcmp('kz', parameter));
            kz = str2num(value{idx});

            idx = find(strcmp('sx', parameter));
            sx = str2num(value{idx});
            idx = find(strcmp('sy', parameter));
            sy = str2num(value{idx});
            idx = find(strcmp('sz', parameter));
            sz = str2num(value{idx});

            %%PC VIPR Orientation
            U_pcvipr = zeros(3, 3);
            U_pcvipr(1, :) = [ix jx kx];
            U_pcvipr(2, :) = [iy jy ky];
            U_pcvipr(3, :) = [iz jz kz];
            O_pcvipr = [sx; sy; sz];

            info.U = U_pcvipr;
            info.O = O_pcvipr;
        end
