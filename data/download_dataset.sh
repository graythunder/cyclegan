#!/bin/bash

DATANAME=$1
if ! `cat datalist.txt | grep -qE ^"${DATANAME}"$`; then 
    echo "Dataset ${DATANAME} is not available. Choose from :";
    cat datalist.txt;
    exit 1;
fi
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/${DATANAME}.zip

wget -N $URL -O ${DATANAME}.zip
unzip ${DATANAME}.zip -d .
rm ${DATANAME}.zip

#mkdir -p ${DATANAME}/{train,test}

