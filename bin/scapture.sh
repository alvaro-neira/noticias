#!/bin/sh

NSECONDS=3540
DATEFULL=$(date '+%Y_%m_%d')
THEHOUR=$(date '+%H')
THEHOUR=${THEHOUR#0}
THEHOUR=$(expr "$THEHOUR" + 1)

if [ "$THEHOUR" -lt 10 ]; then
    FILENAME="/Users/aneira/Desktop/tv24horas_${DATEFULL}_0${THEHOUR}.mov"
else
    FILENAME="/Users/aneira/Desktop/tv24horas_${DATEFULL}_${THEHOUR}.mov"
fi
echo "$FILENAME"
date
echo $NSECONDS

#Main Display
/usr/sbin/screencapture -m -V $NSECONDS -x -a "$FILENAME"

#Secondary Display
#/usr/sbin/screencapture -D 2 -V $NSECONDS -x -a "$FILENAME"
