#!/bin/sh

THEPATH="/Users/aneira/Desktop"

for FILE in "$THEPATH"/tv24horas_*.mp4; do
  echo "$FILE"
  [ -e "$FILE" ] || continue
  /Users/aneira/noticias/bin/upload.py "$FILE" && rm -f "$FILE"
  echo "done with file $FILE"
  exit 0
done
