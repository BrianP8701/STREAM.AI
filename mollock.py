from src import g_parser as parser
from src import drawing as draw
from src import constants as constants
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

video = constants.LOCAL_PATHS.get('8000_path')
gcode = constants.LOCAL_PATHS.get('8000_gcode')
tips, angles, temporary_coordinates, temporal_offsets = parser.tip_tracker(gcode, 30, 14.45, 1108, 370, [120.857,110, 1.8], video, constants.CORRECT_TIMEK.get('8000_timek'))

frames = [pair[0] for pair in temporal_offsets]
temporal_offsets = [pair[1] for pair in temporal_offsets]
slope, intercept = parser.best_fit_line(frames, temporal_offsets)

plt.scatter(frames, temporal_offsets, color='b', label='Points')
plt.xlim(0, max(frames))
plt.ylim(min(temporal_offsets), max(temporal_offsets))
plt.xlabel('Frames')
plt.ylabel('Temporal Offset')
plt.legend()
plt.savefig('99why.jpg', dpi=300)
plt.close()

x = []
slopes = []
stdevs = []

for i in range (3, len(frames)):
    frames_ = frames[:i]
    temporal_offsets_ = temporal_offsets[:i]
    slope, stdev = parser.least_squares_slope_stddev(frames_, temporal_offsets_)
    x.append(frames_[-1])
    slopes.append(slope)
    stdevs.append(stdev)

plt.scatter(x, slopes, color='r', label='Slopes')
plt.xlim(0, max(frames))
plt.ylim(min(slopes), max(slopes))
plt.xlabel('Frames')
plt.ylabel('Slopes')
plt.legend()
plt.savefig('99slopes.jpg', dpi=300)
plt.close()

plt.scatter(x, stdevs, color='g', label='Stdevs')
plt.xlim(0, max(frames))
plt.ylim(min(stdevs), max(stdevs))
plt.xlabel('Frames')
plt.ylabel('Stdevs')
plt.legend()
plt.savefig('99stdevs.jpg', dpi=300)
plt.close()

frame = 0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
video_filename = 'test_V3.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (1920, 1080))
while(frame < 77777):
    bounding_box = parser.crop_in_direction(tips[frame], angles[frame]*180/math.pi)
    image = draw.draw(f'{video}{frame}.jpg', int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
    image = draw.tip_line(image, (angles[frame]), tips[frame])
    cv2.putText(image, str(frame), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(image)
    frame = frame + 1
out.release()
