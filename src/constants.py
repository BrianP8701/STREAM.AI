from types import MappingProxyType

TEST_INPUTS = MappingProxyType({
    0: ['data/video/Run78.mov', 'data/gcode/TEST_8_13_1_08.gcode', 'data/signal/Run78P_0626.txt',],
})

# Acceleration of extruder tip in mm/s^2
ACCELERATION = 57
# Used to adjust calculated time of moves. Adaptively changes for each new video
TIME_K = 1
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

'''
Time-k with 64 acceleraton
[[0.89,-0.041907006091154525],[0.9,-0.05326824146685957],[0.91,-0.06463476797405285],[0.92,-0.07603653576913012],[0.93,-0.08495023058728211],[0.94,-0.09803920138217907],[0.95,],[0.96,],[0.97,1.0],[0.98,],[0.99,],[1,]]


Acceleration with time-k = 1
[[30,],[35,],[40,],[45,],[50,],[54,],[57,-0.18020990584288044],[60,-0.1742582438325061],[64,-0.1671163544104762],[70,-0.15757006151824238],[80, -0.14396774670287626],[90,-0.13345797243861912],[150,-0.09675999495144812],[200,-0.08024936330384355],[300,-0.062024570165097485],[500,-0.0448532670089066],[1000,-0.02813049469782891],[9000,-0.005694383980313408],[90000,-0.0012068341562851102]]
'''