#!/bin/bash

COLOR_PY=./colorIt.py

if [ "$#" -ne 1 ]; then
    echo USAGE: $0 results_folder >&2
    exit 1
fi

dir=$1

for f in "$dir"/data.*
do
    tr ' ' ',' < "$f" | sed 's/,$//' | "$COLOR_PY" - heat > "$f.png"
done

exit 0

