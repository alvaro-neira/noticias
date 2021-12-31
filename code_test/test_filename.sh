#!/bin/sh

THEYEAR=$(date '+%Y')
THEMONTH=$(date '+%m')
THEMONTH=${THEMONTH#0}
THEDAY=$(date '+%d')
THEHOUR=$(date '+%H')
THEHOUR=${THEHOUR#0}
THEHOUR=$(expr "$THEHOUR" + 1)

FILENAME="/Users/aneira/Desktop/tv24horas"
if [ "$THEHOUR" -lt 10 ]; then
    THEHOUR="0${THEHOUR}"
fi
if [ "$THEMONTH" -lt 10 ]; then
    THEMONTH="0${THEMONTH}"
fi
FILENAME="${FILENAME}_${THEYEAR}_${THEMONTH}_${THEDAY}_${THEHOUR}.mov"
echo ""
echo "filename:"
echo "$FILENAME"
date

