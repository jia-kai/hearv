#!/bin/bash -e
# $File: extract_img.sh
# $Date: Sat Jan 03 00:24:00 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

video=$1
start=$2
duration=$3
output=$4

if [ -z "$duration" ]
then
    echo "usage: $0 <video> <start time> <duration> [<output>]"
    exit
fi

[ -z "$output" ] && output=${video%.mov}
mkdir $output

ffmpeg -ss $start -t $duration -i $video -vn -acodec pcm_s16le "${output}.wav"
ffmpeg -ss $start -t $duration -i $video $EXTRA_OPT -f image2 "$output/image-%04d.png"
