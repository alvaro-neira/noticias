#!/bin/sh

export PYTHONPATH="${PYTHONPATH}:/Users/aneira/noticias/"
THEPATH="/Users/aneira/Desktop"
FFMPEG="/opt/homebrew/Cellar/ffmpeg/4.4.1_3/bin/ffmpeg"

number=$(ps aux | grep -v grep | grep -ci ffmpeg)

if [ "$number" -gt 0 ]
    then
        echo "ffmpeg already running";
        exit 1;
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
  ${FFMPEG} -noautorotate -i "$prefilename.$extension" -vf scale=640:360:flags=neighbor -y "$prefilename.mp4" && \
  rm "$prefilename.$extension" && \
  /Users/aneira/noticias/bin/upload.py "$THEPATH/$prefilename.mp4" && \
  rm -f "$prefilename.mp4"
  echo "done with file $prefilename.$extension"
  exit 0
done
