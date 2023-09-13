from src.threads.Tracker import Tracker

video_path = 'data/video/Run78.mov'
gcode_path = 'data/gcode/TEST_8_13_1_08.gcode'
signals_path = 'data/signal/Run78P_0626.txt'
json_path = 'data/metrics/run2.json'

tracker = Tracker(video_path, gcode_path, signals_path, json_path, display_video=True, save_video=False, resolution_percentage=40)
tracker.start()