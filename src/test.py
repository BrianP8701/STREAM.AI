from typing import List, Tuple


def get_next_line(data: List[str], index):
    index += 1
    line_data = [-1, -1, -1, -1]
    while(index < len(data)):
        if(data[index][:2] == 'G1'): 
            line: list = data[index].split(" ")
            breakout = False
            for c in line:
                if(c[:1] == "X"): 
                    breakout = True
                    line_data[0] = float(c[1:])
                elif(c[:1] == "Y"): 
                    breakout = True
                    line_data[1] = float(c[1:])
                elif(c[:1] == "Z"): 
                    breakout = True
                    line_data[2] = float(c[1:])
                elif(c[:1] == "F"): line_data[3] = float(c[1:])
            if breakout: return line_data, index
        index += 1
    return line_data, -1

g_path = "data/gcode3.gcode"
with open(g_path, 'r') as f_gcode:
        data = f_gcode.read()
        data: list = data.split("\n")

g_index = 99 # Line in gcode
line, g_index = get_next_line(data, g_index)


print(line)