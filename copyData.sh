#!/bin/bash
# This script will pull all new or modified files from Isilon and copy them to the CN machines for each directory listed below. (In rsync commands).
#

if [ $# -eq 0 ]; then
        echo "Usage: ./copyData.sh rad_user_name subjectID visitID"
        exit 1
fi

user=$1
rootPath="~/data"
fainLabFolder="FainLab"
subjectID=$2
visitID=$3
contrast=$4
home=pwd


if [[ "$contrast" == "Pre" ]]; then
        prefix=PreContrast
elif [[ "$contrast" == "Post" ]]; then
        prefix=PostContrast
fi


[ -d /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID ] || mkdir -p /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix

#Pull raw data from Isilon to CN scratch for particular subject.
rsync -rlDPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/ipf/$subjectID/mri/$visitID/*.xml /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix
rsync -rlDPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/ipf/$subjectID/mri/$visitID/*.txt /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix
rsync -rlDPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/ipf/$subjectID/mri/$visitID/Ga* /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix

if [[ "$contrast" == "Pre" ]]; then
        grep_output=$(grep -l -m 1 -e "UTE  Pre-contrast (Part 2)" -e "Continuous UTE Pre - Free Breathing" -e "Continuous UTE  Pre-contrast" /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix/* | head -1)
elif [[ "$contrast" == "Post" ]]; then
        grep_output=$(grep -l -m 1 -e "UTE  Post-contrast (Part 2)" -e "Continuous UTE Post - Free Breathing" -e "Continuous UTE  Post-contrast" /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix/* | head -1)
fi
echo $grep_output
pfile=$(basename $grep_output)
echo "THIS IS THE PFILE WE ARE LOOKING FOR..............."
pfile=${pfile:0:8}
echo $pfile
rsync -rlDvvPhu --chmod=ugo=rwx $user@192.250.20.68:"$rootPath"/$fainLabFolder/rawdata/ipf/$subjectID/mri/$visitID/$pfile /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix

cd /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix/

pcvipr_recon_binary -f /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/$prefix/$pfile -export_kdata

cd $home
#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/*
#grep "UTE  Pre-contrast (Part 2)" | "Continuous UTE Pre - Free Breathing" | "Continuous"*"UTE"*"Pre-contrast"* /scratch/scratch_cnxx/ltorres/$subjectID/mri/$visitID/*

#chmod 777 -R /scratch/scratch_cnxx/ltorres/