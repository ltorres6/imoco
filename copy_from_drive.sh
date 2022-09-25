#!/bin/bash
# This script will pull all new or modified files from Isilon and copy them to the CN machines for each directory listed below. (In rsync commands).
#

if [ $# -eq 0 ]; then
    echo "Usage: ./copy_from_drive.sh <subjectID> <visitID> <Contrast> <Study>"
    exit 1
fi
rootPath="~/data"
fainLabFolder="FainLab"
subjectID=$1
visitID=$2
contrast=$3
study=$4
home=$PWD
external_path='/media/ltorres/Seagate Expansion Drive/rawdata'

if [[ "$contrast" == "None" ]]; then
    prefix=""
elif [[ "$contrast" == "Pre" ]]; then
    prefix=pre_contrast
elif [[ "$contrast" == "Post" ]]; then
    prefix=post_contrast
fi

[ -d /home/ltorres/data/rawdata/$study/$subjectID/mri/$visitID/$prefix ] || mkdir -p /home/ltorres/data/rawdata/$study/$subjectID/mri/$visitID/$prefix

#Pull raw data from Isilon to CN scratch for particular subject.
# # rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.xml /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# # rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.txt /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# # rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/Ga* /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.xml /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.txt /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/Ga* /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDPhuv --chmod=ugo=rwx --progress "$external_path"/$study/$subjectID/mri/$visitID/Ga* /home/ltorres/data/rawdata/$study/$subjectID/mri/$visitID/$prefix/
if [[ "$contrast" == "None" ]]; then
    grep_output=$(grep -l -m 1 -e "Free Breathing UTE" -e "Continuous UTE Pre - Free Breathing" /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/* | head -1)
elif [[ "$contrast" == "Pre" ]]; then
    grep_output=$(grep -l -m 1 -e "Free Breathing UTE" -e "UTE  Pre-contrast (Part 2)" -e "Continuous UTE Pre - Free Breathing" -e "Continuous UTE  Pre-contrast" "$external_path"/$study/$subjectID/mri/$visitID/*.txt | head -1)
elif [[ "$contrast" == "Post" ]]; then
    grep_output=$(grep -l -m 1 -e "UTE  Post-contrast (Part 2)" -e "Continuous UTE Post - Free Breathing" -e "Continuous UTE  Post-contrast" /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/* | head -1)
fi
echo $grep_output
pfile=$(basename "$grep_output")
echo "THIS IS THE PFILE WE ARE LOOKING FOR..............."
pfile=${pfile:0:8}
echo $pfile

[[ -z "$pfile" ]] && {
    echo "Pfile is empty. Probably no data."
    exit 1
}

# rsync -rlDvvPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/$pfile /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDvvPhu --chmod=ugo=rwx "$external_path"/$study/$subjectID/mri/$visitID/$pfile /home/ltorres/data/rawdata/$study/$subjectID/mri/$visitID/$prefix/

cd /home/ltorres/data/rawdata/$study/$subjectID/mri/$visitID/$prefix/
echo "Extracting MRI_Raw.h5"
docker run --mount type=bind,source=/home/ltorres/data,target=/data,consistency=cached pcvipr_recon_binary /data/rawdata/$study/$subjectID/mri/$visitID/$prefix/

echo "Removing Pfile and Gating Files"
rm $pfile Ga*
cd $home
# pcvipr_recon_binary -f /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/$pfile -export_kdata

# cd $home

#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /data_local/scratch/ltorres/$subjectID/mri/$visitID/*
#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /data_local/scratch/ltorres/$subjectID/mri/$visitID/*

#chmod 777 -R /data_local/scratch/ltorres/
