#!/bin/bash -e
# $File: try_params.sh
# $Date: Wed Jan 07 02:07:44 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

out='data/params'

mkdir -p $out

function run
{
    export PYR_LEVEL=$1
    export VERT_GROUP_SIZE=$2
    export VERT_GROUP_EXP_DECAY=$3
    dis_cache=""
    [ -z "$PYR_LEVEL" ] || dis_cache="--disable_pyramid_cache"
    basename=$out/$1-$2-$3
    for im0 in 15 16 17
    do
        for im1 in 23 25 32
        do
            oname=$basename-$im0-$im1
            if [ -f "${oname}.txt" ]
            then
                echo "${oname}.txt already exists" > /dev/stderr
                continue
            fi
            ./main.py data/test-song/image-0{$im0,$im1}.png \
                --noise_frame 0 --disp_save ${oname}.png \
                --disable_spectrum_cache $dis_cache \
                --spectrum_snr > ${oname}.txt
        done
    done
    avg=${basename}.avg.txt
    echo  "print ($(paste -sd+ <(cat $basename-*.txt)))/9.0" | python2 \
        > $avg
    cat $avg
}

for vg in $(seq 10 10 300) 400 500 600 700 800
do
    echo "$vg $(run '' $vg)"
done > $out/vg.txt

run '' '' 1 > $out/decay.txt

for lvl in 0 1 2 3
do
    echo "$lvl $(run $lvl)"
done > $out/lvl.txt

for lvl in 0 1 2 3
do
    echo "$lvl $(run $lvl 10)"
done > $out/lvl-vg10.txt
