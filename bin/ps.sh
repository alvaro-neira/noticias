#!/bin/sh

number=$(ps aux | grep -v grep | grep -ci ffmpeg)

if [ $number -gt 0 ]
    then
        echo Running;
    else
        echo "not running"
fi

