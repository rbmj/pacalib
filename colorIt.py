#!/usr/bin/env python

from __future__ import print_function
import Image
import sys
import math
import colorsys
import importlib
import csv

import colorMaps
color_maps = {}
for m in colorMaps.__all__:
    importlib.import_module('colorMaps.'+m).register(color_maps)

def parse_csv(datafile, logarithmic=False):
    reader = csv.reader(datafile)
    data = []
    if logarithmic:
        data = [[math.log(float(x)+1) for x in row] for row in reader]
    else:
        data = [[float(x) for x in row] for row in reader]
    largest = data[0][0]
    for line in data:
        for elem in line:
            if elem > largest:
                largest = elem
    return (data, (len(data[0]), len(data)), largest)

logscale = False
if (len(sys.argv) == 4) and sys.argv[3] == 'log':
    logscale = True
    sys.argv = sys.argv[0:3]

if (len(sys.argv) != 3):
    print("USAGE: " + sys.argv[0] + " pixel_map color_plugin [log]", file=sys.stderr)
    print("Supported color maps:", file=sys.stderr)
    for k in color_maps.keys():
        print("\t" + k, file=sys.stderr)
    sys.exit(1)

fin = sys.stdin

if sys.argv[1] != '-':
    fin = open(sys.argv[1], 'r')

(data, dim, largest) = parse_csv(fin, logscale)

fin.close()

img = Image.new('RGB', dim, "black")
pixels = img.load()

selected_map = sys.argv[2]
if selected_map not in color_maps:
    print("Undefined color map " + selected_map, file=sys.stderr)
    sys.exit(1)

for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixels[i,j] = color_maps[selected_map](data[j][i], largest)

img.save(sys.stdout, "PNG")

