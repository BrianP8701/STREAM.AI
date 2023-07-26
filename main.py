import src.constants as c
from src.threads import main_thread

test_input = c.TEST_INPUTS[0]
main_thread(test_input[0], test_input[1], test_input[2], display_video=True, save_video=False, save_path='out.mov')