from src.threads import main_thread

video_path = 'data/video/Run78.mov'
gcode_path = 'data/gcode/TEST_8_13_1_08.gcode'
signals_path = 'data/signal/Run78P_0626.txt'

main_thread(video_path=video_path, gcode_path=gcode_path, signals_path=signals_path, display_video=True, 
            display_fps=6, save_video=False, save_fps=6, save_path='out.mov', resolution_percentage=40)