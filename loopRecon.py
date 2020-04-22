import os
import subprocess

# loop_moco.py
codeDir = "/export/home/ltorres/projects/xdgrasp"
rootDir = "/scratch/scratch_cnxx/ltorres"
subjectList = os.listdir(rootDir)
subjectList.remove("ipf")
subjectList.sort()
# print(*subjectList, sep=", ")
try:
    for ii in subjectList:
        subject = ii
        if int(subject[4:]) > 19:
            # print(int(subject[4:]))
            visitList = os.listdir(os.path.join(rootDir, subject + "/mri/"))
            visit = visitList[0]
            # print(visit)
            subjectDir = os.path.join(rootDir, subject, "mri", visit)
            print(subjectDir)
            fileList = os.listdir(subjectDir)
            if "MRI_Raw.h5" in fileList:
                "File Exists, Begin!"
            else:
                pass
            # Set up data paths
            h5_path = subjectDir + "/MRI_Raw.h5"
            ksp_path = subjectDir + "/ksp.npy"
            coord_path = subjectDir + "/coord.npy"
            dcf_path = subjectDir + "/dcf.npy"
            kspB_path = subjectDir + "/kspB.npy"
            coordB_path = subjectDir + "/coordB.npy"
            dcfB_path = subjectDir + "/dcfB.npy"
            mps_path = subjectDir + "/mps.npy"
            resp_path = subjectDir + "/resp.npy"
            mrimg_path = subjectDir + "/mrimg"  # no extention for cfl file writing
            # Convert MRI_Raw.h5 to cfl and read resp waveform.
            command = (
                "python "
                + os.path.join(codeDir, "convert_uwute.py")
                + " "
                + h5_path
                + " "
                + ksp_path
                + " "
                + coord_path
                + " "
                + dcf_path
                + " "
                + resp_path
            )
            # os.system(command)
            print("Running File Conversion...")
            # subprocess.call([command], shell=True)
            # AutoFOV to reduce matrix size
            command = (
                "python "
                + os.path.join(codeDir, "autofov.py")
                + " "
                + ksp_path
                + " "
                + coord_path
                + " "
                + dcf_path
                + " "
                + subjectDir  # Diagnostics Directory
                + " --thresh 0.4"
                + " --device 3"
            )
            # os.system(command)
            print("Running Autofov...")
            subprocess.call([command], shell=True)
except KeyboardInterrupt:
    print("interrupted!")
