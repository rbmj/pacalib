import colorsys

def heat_map(x, largest):
    return tuple(int(255*c) 
            for c in colorsys.hls_to_rgb(0.66 + 0.33*((x-10.)/80.), .5, 1))

def register(color_maps):
    color_maps['heat'] = heat_map

