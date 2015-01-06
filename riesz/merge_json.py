#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: merge_json.py
# $Date: Mon Jan 05 21:29:28 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cjson
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='merge json examples for display',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-i', '--input', action='append', required=True,
                        help='input json file')
    parser.add_argument('-n', '--name', action='append', required=True,
                        help='name of the example')
    parser.add_argument('-a', '--audio_link', action='append', required=True,
                        help='audio file path')
    parser.add_argument('-v', '--video_link', action='append', required=True,
                        help='video file path')
    args = parser.parse_args()
    nr = len(args.input)
    assert all(nr == len(i) for i in [args.name, args.audio_link, args.video_link])
    result = {}
    for i in range(nr):
        with open(args.input[i]) as fin:
            data = cjson.decode(fin.read())
        data['video_link'] = args.video_link[i]
        data['audio_link'] = args.audio_link[i]
        result[args.name[i]] = data
    with open(args.output, 'w') as fout:
        fout.write(cjson.encode(result))

if __name__ == '__main__':
    main()
