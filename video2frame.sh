#!/bin/bash

#for folder in */
cd data/train
for folder in {S03,}
do
  cd $folder
  for D in */ #`find folder -type d` 
  do
    cd $D
    pwd
    mkdir images
    ffmpeg  -i vdo.avi images/%04d.png
    cd ..
  done
  cd ..
done
cd ../..

