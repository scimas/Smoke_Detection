#!/bin/bash

if [ ! -d "../data" ]; then
	mkdir ../data
fi

echo "Downloading the data"

wget "https://onedrive.live.com/?authkey=%21AFYQkl1tP%2DQh3Ek&id=2B888FC2F8F47809%21857&cid=2B888FC2F8F47809#authkey=%21AFYQkl1tP%2DQh3Ek&cid=2B888FC2F8F47809&id=2B888FC2F8F47809%21858&parId=2B888FC2F8F47809%21857&action=defaultclick" -O ../data/SmokeData.zip


if [ $? -eq 0 ]
then
    echo "Download Completed"
    cd ../data/
    unzip SmokeData.zip

    if [ $? -eq 0 ]
    then
        echo "The files are sucessfully unzipped"
        echo "The directory has below folders:"
        ls -1 SmokeData/
    else
        echo "error unzipping files"
    fi
else
    echo "Error Downloading the files"
fi
