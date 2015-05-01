import colorsys

def red_map(x, largest):
    return tuple(int(255*c) 
            for c in colorsys.hls_to_rgb(0, float(x)/largest, 1))

def register(color_maps):
    color_maps['red'] = red_map
    
