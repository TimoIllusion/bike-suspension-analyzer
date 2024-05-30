#%%

import os
import math
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from loguru import logger

#TODO: smoothing of plots
#TODO: single click ROIs
#TODO: add executable (with file dialogue, export etc.)

#%%

parser = argparse.ArgumentParser(description='Analyze motobike suspension')
parser.add_argument('video_images_path', type=str, help='Path to video images after extraction')
parser.add_argument('--start', type=int, default=1, help='Start frame number (has to start with 1) (default: 0)')
parser.add_argument('--end', type=int, default=1000, help='End frame number (default: 1000)')
args = parser.parse_args()

assert args.start > 0, "Start frame has to be greater than 0 due to ffmpeg generated frames starting with 00001"


cache_path = ".cache/cache.json"
output_dir = "./output"
result_filename = "results.json"
output_video_filename = "out.mp4"
output_plot_filename = "plot.png"
frame_counter = args.start
end_frame = args.end
fps = 30


result_filepath = os.path.join(output_dir, result_filename)
output_video_path = os.path.join(output_dir, output_video_filename)
output_plot_path =  os.path.join(output_dir, output_plot_filename)

def cached_roi_location_available() -> bool:
    return os.path.exists(cache_path)

def load_cached_roi_location():

    with open(cache_path, "r") as f:
        bboxes = json.load(f)

    return bboxes["bbox_top_left"], bboxes["bbox_top_right"], bboxes["bbox_bottom_left"], bboxes["bbox_bottom_right"]

def manually_set_roi_locations():
    bbox_top_left = cv2.selectROI('bbox_top_left', frame, False)
    bbox_top_right = cv2.selectROI('bbox_top_right', frame, False)
    bbox_bottom_left = cv2.selectROI('bbox_bottom_left', frame, False)
    bbox_bottom_right = cv2.selectROI('bbox_bottom_right', frame, False)

    bboxes = {"bbox_top_left": bbox_top_left, "bbox_top_right": bbox_top_right, "bbox_bottom_left": bbox_bottom_left, "bbox_bottom_right": bbox_bottom_right}

    cache_parent_dir = os.path.dirname(cache_path)
    os.makedirs(cache_parent_dir, exist_ok=True)

    with open(cache_path, "w") as f:
        json.dump(bboxes, f, indent=4)

    return bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right

def try_to_load_roi_cache_otherwise_set_manually():

    if cached_roi_location_available():
        return load_cached_roi_location()
    else:
        return manually_set_roi_locations()
    
class Tracklet:

    def __init__(self, bbox):
        x, y, w, h = [int(i) for i in bbox]
        self.center = (int(x + w/2), int(y + h/2))

    

def euclidean_distance(a: Tracklet, b: Tracklet):
    return math.sqrt(math.pow((b.center[0] - a.center[0]), 2) + math.pow((b.center[1] - a.center[1]), 2))


detections = {}
front_distances = []
front_distances_frame_ids = []
back_distances = []
back_distances_frame_ids = []

initialized = False

os.makedirs("output", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps,  (1280, 720))

while (True):
    
    frame_file = args.video_images_path + "/" + f"{frame_counter:05d}.jpg"
    
    logger.trace(f"Processing frame {frame_counter} from {frame_file}")
    
    if not os.path.exists(frame_file) or frame_counter == end_frame:
        logger.error(f"Frame {frame_file} not found. Break loop.")
        break
    
    frame = cv2.imread(frame_file)

    if not initialized:
        # Select the object to track using the mouse event
        bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right = try_to_load_roi_cache_otherwise_set_manually()

        # Initialize the tracker
        tracker_top_left = cv2.TrackerCSRT_create()
        tracker_top_left.init(frame, bbox_top_left)
        tracker_top_right = cv2.TrackerCSRT_create()
        tracker_top_right.init(frame, bbox_top_right)
        tracker_bottom_left = cv2.TrackerCSRT_create()
        tracker_bottom_left.init(frame, bbox_bottom_left)
        tracker_bottom_right = cv2.TrackerCSRT_create()
        tracker_bottom_right.init(frame, bbox_bottom_right)

        trackers = [tracker_top_left, tracker_top_right, tracker_bottom_left, tracker_bottom_right]

        initialized = True
        cv2.destroyAllWindows()
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL) 

    tracklets = [] # bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right
    success_results = []
    for tracker in trackers:
      
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        tracklets.append(Tracklet(bbox))
        success_results.append(success)

    marker_upper_front = tracklets[1]
    marker_lower_front = tracklets[3]
    
    front_distance = euclidean_distance(marker_upper_front, marker_lower_front)
    cv2.putText(frame, str(np.round(front_distance, 1)), (marker_upper_front.center[0] + 50, marker_upper_front.center[1] + 50), 0, 1.2, (0, 0, 0), 3)
    cv2.line(frame, marker_upper_front.center, marker_lower_front.center, (255, 0, 0), 6)
    front_distances.append(front_distance)
    front_distances_frame_ids.append(frame_counter)
        
    marker_upper_back = tracklets[0]
    marker_lower_back = tracklets[2]
    
    back_distance = euclidean_distance(marker_upper_back, marker_lower_back)
    cv2.putText(frame, str(np.round(back_distance, 1)), (marker_upper_back.center[0] + 50, marker_upper_back.center[1] + 50), 0, 1.2, (0, 0, 0), 3)
    cv2.line(frame, marker_upper_back.center, marker_lower_back.center, (0, 0, 255), 6)
    back_distances.append(back_distance)
    back_distances_frame_ids.append(frame_counter)

    frame = cv2.resize(frame, (1280, 720))
            
    cv2.putText(frame, f"Frame ID {frame_counter}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    writer.write(frame)

    cv2.imshow("Preview", frame)
    cv2.waitKey(1)
      
    frame_counter += 1
    
writer.release()
    
#%%
data = {
    "frame_ids":front_distances_frame_ids,
    "front": front_distances,
    "back": back_distances
}



with open(result_filepath, "w") as f:
    json.dump(data, f, indent=4)

#%%

# front_max = np.max(front_distances)
# back_max = np.max(back_distances)

# global_max = np.max((front_max, back_max))

# front_distances = front_distances / global_max
# back_distances = back_distances / global_max

front_distances = np.array(front_distances) - front_distances[0]
back_distances = np.array(back_distances) - back_distances[0]

plt.figure(figsize=(16, 9))
plt.plot(front_distances_frame_ids, front_distances, color="blue")
plt.plot(front_distances_frame_ids, back_distances, color="r")
plt.title("Tracklet distance progression FRONT and BACK")
plt.xlabel("# Frame")
plt.ylabel("Euclidean distance relative to starting distance in px")
plt.legend(["front", "back"])

plt.savefig(output_plot_path)

