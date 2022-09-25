import logging
import h5py
import numpy as np
import argparse
import os
from ute_recon_tools.coil_compression import pca_cc
import ute_recon_tools.cfl as cfl
from tqdm import trange


def get_cov(noise):
    """Get covariance matrix from noise measurements.

    Args:
        noise (array): Noise measurements of shape [num_coils, ...]

    Returns:
        array: num_coils x num_coils covariance matrix.

    """
    num_coils = noise.shape[0]
    X = noise.reshape([num_coils, -1])
    X -= np.mean(X, axis=-1, keepdims=True)
    cov = np.matmul(X, X.T.conjugate())

    return cov


def whiten(ksp, cov):
    """Whitens k-space measurements.

    Args:
        ksp (array): k-space measurements of shape [num_coils, ...]
        cov (array): num_coils x num_coils covariance matrix.

    Returns:
        array: whitened k-space array.

    """
    num_coils = ksp.shape[0]

    x = ksp.reshape([num_coils, -1])

    L = np.linalg.cholesky(cov)
    # Modified since np.linalg.solve can consume a large amount of memory. Verified with np.all_close()
    n_points = x.shape[-1]
    n_chunks = 50
    points_per_chunk = n_points // n_chunks
    idx = np.arange(0, x.shape[-1], points_per_chunk)
    idx[-1] = n_points

    x_w = np.empty_like(x)
    for chunk in trange(n_chunks - 1, ncols=60, desc="Whitening"):
        x_w[..., idx[chunk] : idx[chunk + 1]] = np.linalg.solve(L, x[..., idx[chunk] : idx[chunk + 1]])
    ksp_w = x_w.reshape(ksp.shape)

    return ksp_w


def convert_ute(h5_file, max_coils=8, dsfSpokes=1.0, compress_coils=False):

    with h5py.File(h5_file, "r") as hf:

        try:
            num_encodes = np.squeeze(hf["Kdata"].attrs["Num_Encodings"])
            num_coils = np.squeeze(hf["Kdata"].attrs["Num_Coils"])
            num_frames = np.squeeze(hf["Kdata"].attrs["Num_Frames"])

            trajectory_type = [
                np.squeeze(hf["Kdata"].attrs["trajectory_typeX"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeY"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeZ"]),
            ]

            dft_needed = [
                np.squeeze(hf["Kdata"].attrs["dft_neededX"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededY"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededZ"]),
            ]

            logging.debug(f"Frames {num_frames}")
            logging.debug(f"Coils {num_coils}")
            logging.debug(f"Encodings {num_encodes}")
            logging.debug(f"Trajectory Type {trajectory_type}")
            logging.debug(f"DFT Needed {dft_needed}")

        except Exception:
            logging.warning("Missing H5 Attributes...")

            num_coils = 0
            while f"KData_E0_C{num_coils}" in hf["Kdata"]:
                num_coils += 1
            logging.debug(f"Number of coils: {num_coils}")

            num_encodes = 0
            while f"KData_E{num_encodes}_C0" in hf["Kdata"]:
                num_encodes += 1
            logging.debug(f"Number of encodes: {num_encodes}")

        # if max_coils is not None:
        #     num_coils = min(max_coils, num_coils)

        coords = []
        dcfs = []
        kdata = []
        ecgs = []
        resps = []

        for encode in range(num_encodes):

            logging.debug(f"Loading encode {encode}")
            # Load timing and resp waveform and sort
            logging.debug(f"Loading timings and resp signal")
            try:
                time = np.squeeze(hf["Gating"][f"time"])
                order = np.argsort(time)
            except Exception:
                time = np.squeeze(hf["Gating"][f"TIME_E{encode}"])
                order = np.argsort(time)

            try:
                resp = np.squeeze(hf["Gating"][f"resp"])
                resp = resp[order]
            except Exception:
                resp = np.squeeze(hf["Gating"][f"RESP_E{encode}"])
                resp = resp[order]

            logging.debug("Loading Coordinates")
            coord = []
            for i in ["Z", "Y", "X"]:
                # logging.debug(f"Loading {i} coords.")
                coord.append(hf["Kdata"][f"K{i}_E{encode}"][0][order])
            coord = np.stack(coord, axis=-1)

            logging.debug("Loading density compensation function")
            dcf = np.array(hf["Kdata"][f"KW_E{encode}"][0][order])

            logging.debug("Loading ECG")
            try:
                ecg = np.squeeze(hf["Gating"][f"ecg"])
                ecg = ecg[order]
            except Exception:
                ecg = np.squeeze(hf["Gating"][f"ECG_E{encode}"])
                ecg = ecg[order]

            # Get k-space
            ksp = []
            for c in range(num_coils):
                logging.debug(f"Loading kspace, coil {c + 1} / {num_coils}.")
                ksp.append(
                    hf["Kdata"][f"KData_E{encode}_C{c}"]["real"][0][order]
                    + 1j * hf["Kdata"][f"KData_E{encode}_C{c}"]["imag"][0][order]
                )
            logging.debug(f"Stacking as np array...")
            ksp = np.stack(ksp, axis=0)
            logging.debug("num_coils {}".format(num_coils))

            try:
                noise = hf["Kdata"]["Noise"]["real"] + 1j * hf["Kdata"]["Noise"]["imag"]
                logging.debug("Whitening ksp.")
                cov = get_cov(noise)
                ksp = whiten(ksp, cov)
            except (MemoryError, Exception) as err:
                logging.warning(f"{err}. Scaling k-space by max value.")
                ksp /= np.abs(ksp).max()

            if compress_coils:
                logging.info("Compressing to {} channels.".format(max_coils))
                ksp = pca_cc(kdata=ksp, axis=0, target_channels=max_coils)

            # Append to list
            coords.append(coord)
            dcfs.append(dcf)
            kdata.append(ksp)
            ecgs.append(ecg)
            resps.append(resp)

            # Log the data
            logging.debug(f"MRI coords encode shape: {coords[encode].shape}")
            logging.debug(f"MRI dcf encode shape: {dcfs[encode].shape}")
            logging.debug(f"MRI kdata encode shape: {kdata[encode].shape}")
            logging.debug(f"MRI ecg encode shape: {ecgs[encode].shape}")
            logging.debug(f"MRI resp encode shape: {resps[encode].shape}")

    # Stack the data along projections (no reason not to keep encodes separate in my case)
    kdata = np.concatenate(kdata, axis=1)
    dcfs = np.concatenate(dcfs, axis=0)
    coords = np.concatenate(coords, axis=0)
    ecgs = np.concatenate(ecgs, axis=0)
    resps = np.concatenate(resps, axis=0)

    # crop empty calibration region (1800 spokes for ipf)
    # kdata = kdata[:, :-1800, :]
    # coords = coords[:-1800, :, :]
    # dcfs = dcfs[:-1800, :]
    # resps = resps[:-1800]
    # ecgs = ecgs[:-1800]

    # crop to desired number of spokes (all by default)
    totalSpokes = kdata.shape[1]
    nSpokes = int(totalSpokes // dsfSpokes)
    kdata = kdata[:, :nSpokes, :]
    coords = coords[:nSpokes, :, :]
    dcfs = dcfs[:nSpokes, :]
    resps = resps[:nSpokes]
    logging.debug(f"Total Number of Spokes: {totalSpokes}, Requested Number of Spokes: {nSpokes}")

    # Log the data
    logging.debug(f"MRI coords final shape: {coords.shape}")
    logging.debug(f"MRI dcf final shape: {dcfs.shape}")
    logging.info(f"MRI kdata final shape: {kdata.shape}")
    logging.debug(f"MRI ecg final shape: {ecgs.shape}")
    logging.debug(f"MRI resp final shape: {resps.shape}")

    # Get TR
    d_time = time[order]
    tr = d_time[1] - d_time[0]

    return kdata, coords, dcfs, resps / resps.max(), tr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converts UWUTE h5 files to npy arrays in natural time ordering.")
    parser.add_argument("h5_file", type=str)
    parser.add_argument("ksp_file", type=str)
    parser.add_argument("coord_file", type=str)
    parser.add_argument("dcf_file", type=str)
    parser.add_argument("resp_file", type=str)
    parser.add_argument("tr_file", type=str)
    parser.add_argument("--max_coils", type=int, default=8)
    parser.add_argument("--dsf_spokes", type=float, default=1.0)
    parser.add_argument("--compress_coils", default=False, action="store_true")
    parser.add_argument("--save_type", type=str, default="numpy")  # 0 = numpy, 1 = cfl, 2 = hdf5, etc...
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ksp, coord, dcf, resp, tr = convert_ute(
        args.h5_file, max_coils=args.max_coils, dsfSpokes=args.dsf_spokes, compress_coils=args.compress_coils
    )
    logging.debug("Saving data.")
    for filename in [args.ksp_file, args.coord_file, args.dcf_file, args.resp_file, args.tr_file]:
        try:
            os.remove(filename)
        except Exception as e:
            logging.warning(e)
            pass
    # if os.path.isfile(args.ksp_file):
    #     os.remove(args.ksp_file)
    #     os.remove(args.coord_file)
    #     os.remove(args.dcf_file)
    #     os.remove(args.resp_file)
    #     os.remove(args.tr_file)

    if args.save_type in "numpy":
        "Saving as numpy format!"
        np.save(args.ksp_file, ksp)
        np.save(args.coord_file, coord)
        np.save(args.dcf_file, dcf)
        np.save(args.resp_file, resp)
        np.savetxt(args.tr_file, np.atleast_1d(tr))
    elif args.save_type in "cfl":
        "Saving as cfl format!"
        cfl.writecfl(args.ksp_file, ksp)
        cfl.writecfl(args.coord_file, coord)
        cfl.writecfl(args.dcf_file, dcf)
        np.savetxt(args.resp_file, resp)
        np.savetxt(args.tr_file, np.atleast_1d(tr))
    else:
        print("Dont understand which format to save in.")
