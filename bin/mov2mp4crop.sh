#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:/Users/aneira/noticias/"
THEPATH="/Users/aneira/Desktop"
FFMPEG="/usr/local/bin/ffmpeg"

number=$(ps aux | grep -v grep | grep -ci ffmpeg)

if [ "$number" -gt 0 ]; then
  echo "ffmpeg already running"
  exit 1
fi

echo $THEPATH
for FILE in "$THEPATH"/*.mov; do
  echo "$FILE"
  [ -e "$FILE" ] || continue
  filename="${FILE##*/}"
  extension="${filename##*.}"
  prefilename="${filename%.*}"
  echo "$prefilename"
  echo "$extension"
  echo ""
  cd $THEPATH
  ${FFMPEG} -noautorotate -i "$prefilename.$extension" -vf scale=640:400:flags=neighbor -y uncropped.mp4 && \
  ${FFMPEG} -noautorotate -i uncropped.mp4 -filter:v "crop=640:360:0:20" -c:a copy -y "$prefilename.mp4" && \
  rm "$prefilename.$extension" && \
  rm uncropped.mp4 && \
  /Users/aneira/noticias/bin/upload.py "$THEPATH/$prefilename.mp4" && \
  rm -f "$prefilename.mp4"
  echo "done with file $prefilename.$extension"
  exit 0
done
