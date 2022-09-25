#!/bin/bash
# This script will pull all new or modified files from Isilon and copy them to the CN machines for each directory listed below. (In rsync commands).
#

if [ $# -eq 0 ]; then
    echo "Usage: ./copyData.sh <rad_user_name> <subjectID> <visitID> <Contrast> <Study>"
    exit 1
fi

user=$1
rootPath="~/data"
fainLabFolder="FainLab"
subjectID=$2
visitID=$3
contrast=$4
study=$5
home=$PWD

if [[ "$contrast" == "None" ]]; then
    prefix=""
elif [[ "$contrast" == "Pre" ]]; then
    prefix=PreContrast
elif [[ "$contrast" == "Post" ]]; then
    prefix=PostContrast
fi

[ -d /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix ] || mkdir -p /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix

#Pull raw data from Isilon to CN scratch for particular subject.
# rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.xml /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.txt /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
# rsync -rlDPhuv --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/Ga* /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.xml /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/*.txt /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDPhuv --chmod=ugo=rwx --progress -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/Ga* /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
if [[ "$contrast" == "None" ]]; then
    grep_output=$(grep -l -m 1 -e "Free Breathing UTE" -e "Continuous UTE Pre - Free Breathing" /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/* | head -1)
elif [[ "$contrast" == "Pre" ]]; then
    grep_output=$(grep -l -m 1 -e "Free Breathing UTE" -e "UTE  Pre-contrast (Part 2)" -e "Continuous UTE Pre - Free Breathing" -e "Continuous UTE  Pre-contrast" /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/* | head -1)
elif [[ "$contrast" == "Post" ]]; then
    grep_output=$(grep -l -m 1 -e "UTE  Post-contrast (Part 2)" -e "Continuous UTE Post - Free Breathing" -e "Continuous UTE  Post-contrast" /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/* | head -1)
fi
echo $grep_output
pfile=$(basename $grep_output)
echo "THIS IS THE PFILE WE ARE LOOKING FOR..............."
pfile=${pfile:0:8}
echo $pfile

[[ -z "$pfile" ]] && {
    echo "Pfile is empty. Probably no data."
    exit 1
}

# rsync -rlDvvPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/$pfile /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix
rsync -rlDvvPhu --chmod=ugo=rwx -e 'ssh -A -J sdc@192.250.20.170' $user@10.151.52.225:"$rootPath"/$fainLabFolder/rawdata/$study/$subjectID/mri/$visitID/$pfile /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix

cd /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/

pcvipr_recon_binary -f /data_local/scratch/ltorres/$study/$subjectID/mri/$visitID/$prefix/$pfile -export_kdata

cd $home

#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /data_local/scratch/ltorres/$subjectID/mri/$visitID/*
#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /data_local/scratch/ltorres/$subjectID/mri/$visitID/*

#chmod 777 -R /data_local/scratch/ltorres/
