#!/bin/bash

if [[ $# -eq 0 ]] ; then
	echo "Specify the directory"
	exit 1
fi

cd $1

if [[ $? -ne 0 ]]; then
	echo "Failed to access specified directory"
	exit 1
fi

echo "I will create a video from the images contained in:"
echo $(pwd) 

while true; do
    read -p "Is this correct? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done



ls -v | cat -n | while read i f; do mv "$f" `printf "%05d.bmp" "$i"`; done && ffmpeg -framerate 60 -pattern_type glob -i "*.bmp" -c:v libx264 -pix_fmt yuv420p ../output.mp4