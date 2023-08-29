# YOLOHandler.py
import threading
import src.threads.global_vars as GV
import src.helpers.helper_functions as helpers
import src.threads.constants as c

class ErrorCorrection:
    def __init__(self):
        pass
    
    def temporal_thread(self, frame, signal_index):
        if signal_index < len(GV.corner_indices): 
            predicted_time_frame = GV.corner_indices[signal_index]
        else: return
         
        screen_temporal_offset = frame - predicted_time_frame
        GV.screen_predictions, GV.bed_predictions, GV.angles, GV.corner_indices = helpers.time_travel(GV.screen_predictions, GV.bed_predictions, GV.angles, GV.corner_indices, frame, screen_temporal_offset)

    def spatial_thread(self, frame, real_screen_tip):
        spatial_offset = [real_screen_tip[0] - GV.screen_predictions[frame][0], real_screen_tip[1] - GV.screen_predictions[frame][1]]
        GV.x_spatial_offsets.append([frame, spatial_offset[0]])
        GV.y_spatial_offsets.append([frame, spatial_offset[1]])
        
        if len(GV.x_spatial_offsets) > c.X_SPATIAL_MIN_SIGNALS: GV.x_spatial_offsets.pop(0)
        if len(GV.y_spatial_offsets) > c.Y_SPATIAL_MIN_SIGNALS: GV.y_spatial_offsets.pop(0)
        
        # x spatial offsets
        if len(GV.x_spatial_offsets) == c.X_SPATIAL_MIN_SIGNALS: # If we have enough signals
            if max(GV.x_spatial_offsets, key=lambda x: x[1])[1] - min(GV.x_spatial_offsets, key=lambda x: x[1])[1] < c.X_SPATIAL_MAX_DEVIATION: # If the spatial offsets are close enough together
                average_spatial_offset = sum(pair[1] for pair in GV.x_spatial_offsets) / len(GV.x_spatial_offsets)
                if abs(average_spatial_offset) > 2:
                    helpers.print_text('Adjust X spatially', 'green')
                    for screen_prediction in GV.screen_predictions[frame:]:
                        screen_prediction[0] += average_spatial_offset  
                    GV.x_spatial_offsets.clear()
        
        # y spatial offsets
        if len(GV.y_spatial_offsets) == c.Y_SPATIAL_MIN_SIGNALS: # If we have enough signals
            if max(GV.y_spatial_offsets, key=lambda x: x[1])[1] - min(GV.y_spatial_offsets, key=lambda x: x[1])[1] < c.Y_SPATIAL_MAX_DEVIATION: # If the spatial offsets are close enough together
                average_spatial_offset = sum(pair[1] for pair in GV.y_spatial_offsets) / len(GV.y_spatial_offsets)
                if abs(average_spatial_offset) > 10: # If the spatial offset is large enough to be significant
                    helpers.print_text('Adjust Y spatially', 'green')
                    for screen_prediction in GV.screen_predictions[frame:]:
                        screen_prediction[1] += average_spatial_offset
                    GV.current_y += average_spatial_offset
                    GV.y_spatial_offsets.clear()       