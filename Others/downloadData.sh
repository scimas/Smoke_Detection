#!/bin/bash

if [ ! -d "../data" ]; then
	mkdir ../data
fi

echo "Downloading the data"

wget "https://ieqepw.by.files.1drv.com/y4mBVaHLVIHDjY09o07uLRZmexNgZuE8dkV3Xx9F4KMBDaiH4maWXYYf147972_oCOCILJ-SES8mS-rF_ui9OescSpvN_6Xl5vFXH-nfxvqn9oYzOts8eQvZJIOiyiU1APBPoLcAGzDqz8grr0sMCq22QYrmbmOE9BdenitNKBL-DYTtcVOiUNcHEevR5EEcyvVKSTy3yFNr1nP71CX3iL2wQ" -O ../data/SmokeData.zip


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
