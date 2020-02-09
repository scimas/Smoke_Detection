#!/bin/bash

if [ ! -d "../data" ]; then
	mkdir ../data
fi


wget -bqco download_log.log "https://onedrive.live.com/?authkey=%21AFYQkl1tP%2DQh3Ek&id=2B888FC2F8F47809%21857&cid=2B888FC2F8F47809#authkey=%21AFYQkl1tP%2DQh3Ek&cid=2B888FC2F8F47809&id=2B888FC2F8F47809%21858&parId=2B888FC2F8F47809%21857&action=defaultclick" -O ../data/SmokeData.zip


echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
echo "wget output is being directed to download_log.log"
echo "use 'tail -f download_log.log' to check progress"

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
