from src import g_parser as parser
from src import drawing as draw
from src import constants as constants
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

video = constants.LOCAL_PATHS.get('video_2_path')
gcode = constants.LOCAL_PATHS.get('gcode2')
tips, angles, temporary_coordinates, temporal_offsets = parser.tip_tracker(gcode, 30, 14.45, 1108, 370, [120.857,110, 1.8], video, constants.TIME_K)

# Stitch frames together into a video
frame = 0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
out = cv2.VideoWriter(video, fourcc, fps, (1920, 1080))
while(frame < 77777):
    bounding_box = parser.crop_in_direction(tips[frame], angles[frame]*180/math.pi)
    image = draw.draw(f'{video}{frame}.jpg', int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
    image = draw.tip_line(image, (angles[frame]), tips[frame])
    cv2.putText(image, str(int(angles[frame]*180/math.pi)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(image)
    frame = frame + 1
    print(frame)
out.release()
