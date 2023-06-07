from . import tip_tracker as parser
from src import drawing as draw
from src import constants as constants
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

args = constants.ARGS.get('2144_args')
video = constants.LOCAL_PATHS.get('2144_path')
tips, angles, temporary_coordinates, temporal_offsets = parser.track(args[0], 30, args[2], args[3], args[4], args[5], args[6], args[7], args[8])

# Plotting, playing with data
frames = [pair[0] for pair in temporal_offsets]
temporal_offsets = [pair[1] for pair in temporal_offsets]
slope, intercept = parser.best_fit_line(frames, temporal_offsets)
plt.scatter(frames, temporal_offsets, color='b', label='Points')
plt.xlim(0, 7300)
plt.ylim(min(temporal_offsets), max(temporal_offsets))
plt.xlabel('Frames')
plt.ylabel('Temporal Offset')
plt.legend()
plt.savefig('99A.jpg', dpi=300)
plt.close()

frame = 0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
video_filename = 'test_V.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (1920, 1080))
while(frame < 77777):
    bounding_box = parser.crop_in_direction(tips[frame], angles[frame]*180/math.pi)
    image = draw.draw(f'{video}{frame}.jpg', int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
    image = draw.tip_line(image, (angles[frame]), tips[frame])
    cv2.putText(image, str(int(angles[frame]*180/math.pi)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(image)
    frame = frame + 1
    print(frame)
out.release()