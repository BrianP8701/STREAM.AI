from types import MappingProxyType

LOCAL_PATHS = MappingProxyType({
    '2073_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2073/frame',
    '8000_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/8000/frame',
    '2061_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2061/frame',
    '2072_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2072/frame',
    '2144_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2144/frame',
    '2143_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2143/frame',
    '2226_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2226/frame',
    '2221_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2221/frame',
    '2107_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2107/frame',
    '2104_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2104/frame',
    '2071_path' : '/Users/brianprzezdziecki/Research/Mechatronics/My_code/VIDEO/2071/frame',
    '2073_gcode' : "data/gcode1.gcode", # Line 0
    '2072_gcode' : "data/gcode1.gcode", # Line 0
    '8000_gcode' : "data/gcode2.gcode", # Line 0
    '2061_gcode' : "data/gcode1.gcode", # Start from line 65
    '2144_gcode' : "data/gcode3.gcode", # Start from line 44
    '2143_gcode' : "data/gcode4.gcode", # Start from line 97
    '2226_gcode' : "data/gcode4.gcode", # Start from line 51
    '2221_gcode' : "data/gcode1.gcode", # Start from line 27 
    '2107_gcode' : "data/gcode5.gcode", # Start from line 58
    '2104_gcode' : "data/gcode3.gcode", # Start from line 99
    '2071_gcode' : "data/gcode1.gcode", # Line 0
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

# G1 X109.2 Y90.8 E35.04896

ARGS = MappingProxyType({
    '2073_args' : [LOCAL_PATHS.get('2073_gcode'), 30, 15.45, 425, 405, [82.554, 82.099, 1.8], LOCAL_PATHS.get('2073_path'), CORRECT_TIMEK.get('2073_timek'), -1],   # Done!
    '8000_args' : [LOCAL_PATHS.get('8000_gcode'), 30, 14.45, 1108, 370, [120.857,110, 1.8], LOCAL_PATHS.get('8000_path'), CORRECT_TIMEK.get('8000_timek'), -1],       # Done!
    '2221_args' : [LOCAL_PATHS.get('2221_gcode'), 30, 12.45, 788, 676, [79.143, 90, 1.8], LOCAL_PATHS.get('2221_path'), CORRECT_TIMEK.get('2221_timek'), 26],        # Done!
    '2061_args' : [LOCAL_PATHS.get('2061_gcode'), 30, 15.6, 943, 607, [109.2, 90.8, 1.8], LOCAL_PATHS.get('2061_path'), CORRECT_TIMEK.get('2061_timek'), 63],        # Done!
    '2144_args' : [LOCAL_PATHS.get('2144_gcode'), 30, 12.55, 987, 589, [120.857,110, 1.8], LOCAL_PATHS.get('2144_path'), CORRECT_TIMEK.get('2144_timek'), 42],        # Done!
    '2072_args' : [LOCAL_PATHS.get('2072_gcode'), 30, 17.1, 467, 796, [82.554, 82.099, 1.8], LOCAL_PATHS.get('2072_path'), CORRECT_TIMEK.get('2072_timek'), -1],    # Done!
    '2071_args' : [LOCAL_PATHS.get('2071_gcode'), 30, 10.5, 658, 699, [82.554, 82.099, 1.8], LOCAL_PATHS.get('2071_path'), CORRECT_TIMEK.get('2071_timek'), -1],    # Done!
    '2104_args' : [LOCAL_PATHS.get('2104_gcode'), 30, 14.45, 447, 694, [90.8, 90.926, 1.8], LOCAL_PATHS.get('2104_path'), CORRECT_TIMEK.get('2104_timek'), 97],
    '2143_args' : [LOCAL_PATHS.get('2143_gcode'), 30, 14.45, 602, 604, [90.8, 90.8, 1.8], LOCAL_PATHS.get('2143_path'), CORRECT_TIMEK.get('2143_timek'), 96],
    '2226_args' : [LOCAL_PATHS.get('2226_gcode'), 30, 14.45, 858, 440, [90, 120.857, 1.8], LOCAL_PATHS.get('2226_path'), CORRECT_TIMEK.get('2226_timek'), 50],
    '2107_args' : [LOCAL_PATHS.get('2107_gcode'), 30, 14.45, 464, 231, [79.143, 90, 1.8], LOCAL_PATHS.get('2107_path'), CORRECT_TIMEK.get('2107_timek'), 57],
})

# Acceleration of extruder tip in mm/s^2
ACCELERATION = 64
# Used to adjust calculated time of moves. Adaptively changes for each new video
TIME_K = 0.992
# Pathname to weights of yolov8 model for correction
YOLO_PATH = 'inference_/best.onnx'
# Inference rates, in frames
MIN_THREAD_ALARM = 120
MAX_THREAD_ALARM = 300    
MAX_SPEED_FOR_STANDARD_HORIZONTAL = 2800
STANDARD_HORIZONTAL_CAP = 60
PRE_VERTICAL_CAP = 210
ACCEPTABLE_RESIDUAL = 15



ACCEPTABLE_PRE_VERTICAL_SPATIAL_ERROR = 15
# mTp        Acceptable_PVSE
# 14.45            20.5




# Whats a low standard deviation?
# High is like above 0.02
# Low is below 0.01
