from measure import g_parser as parser
from measure import Drawing as draw
from measure import constants as constants
import cv2
import math

# Video 1
tips, angles = parser.tip_tracker("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/gcode1.gcode", 30, 15.45212638, 425, 405, [82.554, 82.099, 1.8], '/Users/brianprzezdziecki/Research/Mechatronics/data1/frame', constants.ACCELERATION)
# Video 2
#tips, angles = tip_tracker("/Users/brianprzezdziecki/Research/Mechatronics/STREAM_AI/data/gcode2.gcode", 30, 14.45, 1108, 370, [120.857,110, 1.8], '/Users/brianprzezdziecki/Research/Mechatronics/data2/frame')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 30
video_filename = 'test_V3.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (1920, 1080))

frame = 0

while(frame < 7328):
    bounding_box = parser.crop_in_direction(tips[frame], angles[frame]*180/math.pi)
    image = draw.draw('/Users/brianprzezdziecki/Research/Mechatronics/data1/frame' + str(frame) + '.jpg', int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3]))
    image = draw.tip_line(image, angles[frame], tips[frame])
    cv2.putText(image, str(int(angles[frame]*180/math.pi)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    out.write(image)
    frame = frame + 1
    print(frame)

out.release()

# frame = 6764
# draw_rectangle('/Users/brianprzezdziecki/Research/Mechatronics/data/frame' + str(frame) + '.jpg', int(tips[frame][0]-5), int(tips[frame][1]-5), int(tips[frame][0]+5), int(tips[frame][1]+5))

print(len(tips))