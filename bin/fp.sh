#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Use <input> <output>" >&2
    exit 2
fi
time ffmpeg -i "$1" -s 640x360 -sws_flags neighbor "$2" && rm "$1"
echo done "ffmpeg -i $1 -s 640x360 -sws_flags neighbor $2 && rm $1"

