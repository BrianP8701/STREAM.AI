import src.constants as c
from src.threads import main_thread

test_input = c.TEST_INPUTS[1]
main_thread(test_input[0], test_input[1], test_input[2], display_video=False, save_video=True, save_path='out.mov')