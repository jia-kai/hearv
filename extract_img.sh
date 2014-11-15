#!/bin/bash -e
# $File: extract_img.sh
# $Date: Sat Nov 15 16:56:59 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

video=$1
start=$2
duration=$3

if [ -z "$duration" ]
then
    echo "usage: $0 <video> <start time> <duration>"
    exit
fi

output=${video%.mov}
mkdir $output

ffmpeg -ss $start -t $duration -i $video -vn -acodec pcm_s16le "${output}.wav"
ffmpeg -ss $start -t $duration -i $video -f image2 "$output/image-%03d.png"
