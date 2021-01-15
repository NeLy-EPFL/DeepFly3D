#!/bin/bash

input="folders.txt"
while IFS= read -r folder
do
	df3d-cli -vv -o -r -n 100 $folder --output-folder df3d
done < "$input"
