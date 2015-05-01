colors = {
        -3: (153,0,0),
        -2: (204,0,0),
        -1: (255,0,0),
        0 : (255,255,255),
        1:  (0,204,0),
        2:  (0,153,0),
        3:  (0,102,0),
}

def register(color_map):
    color_map['forrestfire'] = lambda x, _: colors[x]
