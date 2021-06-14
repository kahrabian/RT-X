#!/bin/bash

DS=GITHUB-SE-1Y-REPO_I
SD=(128 128 128 128 96 96 96 96  64 64 64 64  32 32 32 32 )
AD=(0   0   0   0   16 16 16 16  32 32 32 32  48 48 48 48 )
RD=(0   32  64  128 0  32 64 128 0  32 64 128 0  32 64 128)

for i in $(seq 0 15); do
    cp run.template.sh run.${DS}.${i}.sh
    sed -i "s/DS/${DS}/" run.${DS}.${i}.sh
    sed -i "s/SD/${SD[i]}/" run.${DS}.${i}.sh
    sed -i "s/AD/${AD[i]}/" run.${DS}.${i}.sh
    sed -i "s/RD/${RD[i]}/" run.${DS}.${i}.sh
    JOB_ID=i python run.${DS}.${i}.sh
#    rm run.${DS}.${i}.sh
done
