RATIO_INITIALIZATION_MIN_RANGE=5
ACCEPTABLE_Y_DIFFERENCE=30
# Acceleration of extruder tip in mm/s^2
ACCELERATION = 6000
# Used to adjust calculated time of moves. Adaptively changes for each new video
TIME_K = 1
YOLO_PATH = 'src/YOLOv8/best2.onnx'
MOBILE_PATH = 'src/MobileNetv3/mob_l_gmms2_finetune.pt'

# For temporal error correction
TIME_TRAVEL_MIN_SIGNALS=5
TIME_TRAVEL_MAX_DEVIATION=5 # in frames

# For acceleration correction
ACCELERATION_SAMPLE_INTERVAL=10
ACCELERATION_MIN_STDEV=1
ACCELERATION_MIN_SIGNALS=20

# For spatial error correction
X_SPATIAL_MIN_SIGNALS=15
X_SPATIAL_MAX_DEVIATION=10
Y_SPATIAL_MIN_SIGNALS=5
Y_SPATIAL_MAX_DEVIATION=10

# For slope change detection
SLOPE_CHANGE_THRESHOLD=0.1
RECENT_SLOPE_RANGE=1000
RECENT_SLOPE_SAMPLE_INTERVAL=100
SLOPE_CHANGE_SENSITIVITY=0.03