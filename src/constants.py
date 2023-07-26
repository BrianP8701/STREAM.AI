from types import MappingProxyType

TEST_INPUTS = MappingProxyType({
    0: ['data/video/Run78.mov', 'data/gcode/TEST_8_13_1_08.gcode', 'data/signal/Run78P_0626.txt',],
})

CORRECT_TIMEK = MappingProxyType({
    '2073_timek': 0.9855,
    '8000_timek': 0.982,
    '2221_timek': 0.9855,
    '2061_timek': 0.9855,
    '2144_timek': 0.982,
    '2072_timek': 0.9855,
    '2071_timek': 0.9855,
    '2104_timek': 0.982,
    '2143_timek': 0.9855,
    '2226_timek': 0.9855,
    '2107_timek': 0.982,
})

# Acceleration of extruder tip in mm/s^2
ACCELERATION = 8000
# Used to adjust calculated time of moves. Adaptively changes for each new video
TIME_K = 0.992
YOLO_PATH = 'src/YOLOv8/best1.onnx'
MOBILE_PATH = 'src/MobileNetv3/mob_l_gmms2_finetune.pt'

TIME_TRAVEL_MIN_SIGNALS=5
TIME_TRAVEL_MAX_DEVIATION=5 # in frames

ACCELERATION_MIN_SIGNALS=10
ACCELERATION_MIN_STDEV=1

X_SPATIAL_MIN_SIGNALS=15
X_SPATIAL_MAX_DEVIATION=10

Y_SPATIAL_MIN_SIGNALS=5
Y_SPATIAL_MAX_DEVIATION=10
# [[70, 1], ]