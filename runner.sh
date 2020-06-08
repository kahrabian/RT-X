#!/bin/bash

DS=(GITHUB-SE-1M-NODE_I GITHUB-SE-1M-NODE_E GITHUB-SE-1Y-REPO_I GITHUB-SE-1Y-REPO_E)
SD=(128 96 32  96)
AD=(0   32 48  16)
RD=(64  64 32  32)

for i in $(seq 0 5); do
    cp run.template.sh run.${DS[i]}.${i}.sh
    sed -i "s/DS/${DS[i]}/" run.${DS[i]}.${i}.sh
    sed -i "s/SD/${SD[i]}/" run.${DS[i]}.${i}.sh
    sed -i "s/AD/${AD[i]}/" run.${DS[i]}.${i}.sh
    sed -i "s/RD/${RD[i]}/" run.${DS[i]}.${i}.sh
    JOB_ID=i python run.${DS[i]}.${i}.sh
    rm run.${DS[i]}.${i}.sh
done
