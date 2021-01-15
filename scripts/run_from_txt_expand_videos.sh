#!/bin/bash

input="folders.txt"
while IFS= read -r folder_root
do
        find "$folder_root" -type d -name "images" -print0 | while read -d $'\0' folder
        do
            echo $file
            # expand videos
            # vframes 100
            # https://unix.stackexchange.com/a/36363
            for ((i=0; i<7; i++)); do
                ffmpeg -i ""$folder"/camera_"$i".mp4" -qscale:v 2 -start_number 0 ""$folder"/camera_"$i"_img_%d.jpg"  < /dev/null
             done

            # -n 100
            # run df3d
	        CUDA_VISIBLE_DEVICES=1 df3d-cli -vv -o $folder --output-folder df3d

            # delete videos
            for ((i=0; i<7; i++)); do
                find "$folder" -name "*.jpg" -maxdepth 1  -delete
            done
        done
done < "$input"
