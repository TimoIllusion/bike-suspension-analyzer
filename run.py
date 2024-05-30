import os
import math
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger

# Constants
CACHE_FILENAME = "roi_cache.json"
OUTPUT_DIR = "./output"
RESULT_FILENAME = "results.json"
OUTPUT_VIDEO_FILENAME = "out.mp4"
OUTPUT_PLOT_FILENAME = "plot.png"
FPS = 30

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze motobike suspension')
    parser.add_argument('video_images_path', type=str, help='Path to video images after extraction')
    parser.add_argument('--start', type=int, default=1, help='Start frame number (has to start with 1) (default: 1)')
    parser.add_argument('--end', type=int, default=None, help='End frame number (default: None)')
    args = parser.parse_args()

    assert args.start > 0, "Start frame has to be greater than 0 due to ffmpeg generated frames starting with 00001"
    
    if args.end is None:
        args.end = get_image_count(args.video_images_path)
        logger.info(f"End frame not specified. Setting to {args.end}")
    
    return args

# Utility Functions
def get_image_count(path):
    return len([f for f in os.listdir(path) if f.endswith('.jpg') and f != CACHE_FILENAME])

def get_cache_path(video_images_path):
    return os.path.join(video_images_path, CACHE_FILENAME)

# Cache Functions
def cached_roi_location_available(cache_path) -> bool:
    return os.path.exists(cache_path)

def load_cached_roi_location(cache_path):
    with open(cache_path, "r") as f:
        bboxes = json.load(f)
    return bboxes["bbox_top_left"], bboxes["bbox_top_right"], bboxes["bbox_bottom_left"], bboxes["bbox_bottom_right"]

def manually_set_roi_locations(frame, cache_path):
    bbox_top_left = cv2.selectROI('bbox_top_left', frame, True)
    bbox_bottom_left = cv2.selectROI('bbox_bottom_left', frame, True)
    bbox_top_right = cv2.selectROI('bbox_top_right', frame, True)
    bbox_bottom_right = cv2.selectROI('bbox_bottom_right', frame, True)

    bboxes = {"bbox_top_left": bbox_top_left, "bbox_top_right": bbox_top_right, "bbox_bottom_left": bbox_bottom_left, "bbox_bottom_right": bbox_bottom_right}

    cache_parent_dir = os.path.dirname(cache_path)
    os.makedirs(cache_parent_dir, exist_ok=True)

    with open(cache_path, "w") as f:
        json.dump(bboxes, f, indent=4)

    return bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right

def try_to_load_roi_cache_otherwise_set_manually(frame, cache_path):
    if cached_roi_location_available(cache_path):
        return load_cached_roi_location(cache_path)
    else:
        return manually_set_roi_locations(frame, cache_path)

# Tracklet Class
class Tracklet:
    def __init__(self, bbox):
        x, y, w, h = [int(i) for i in bbox]
        self.center = (int(x + w / 2), int(y + h / 2))

def euclidean_distance(a: Tracklet, b: Tracklet) -> float:
    return math.sqrt(math.pow((b.center[0] - a.center[0]), 2) + math.pow((b.center[1] - a.center[1]), 2))

# Main Processing Function
def process_video(args):
    frame_id = args.start
    end_frame = args.end
    cache_path = get_cache_path(args.video_images_path)

    result_filepath = os.path.join(OUTPUT_DIR, RESULT_FILENAME)
    output_video_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILENAME)
    output_plot_path = os.path.join(OUTPUT_DIR, OUTPUT_PLOT_FILENAME)

    front_distances = []
    front_distances_frame_ids = []
    back_distances = []
    back_distances_frame_ids = []

    initialized = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (1280, 720))

    for frame_id in range(args.start, end_frame + 1):
        
        frame_file = os.path.join(args.video_images_path, f"{frame_id:05d}.jpg")
        logger.debug(f"Processing frame {frame_id} from {frame_file}")

        if not os.path.exists(frame_file):
            logger.error(f"Frame {frame_file} not found. Skipping frame.")
            continue
        
        frame = cv2.imread(frame_file)
        frame = cv2.resize(frame, (1280, 720))

        if not initialized:
            bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right = try_to_load_roi_cache_otherwise_set_manually(frame, cache_path)

            trackers = initialize_trackers(frame, [bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right])
            initialized = True
            cv2.destroyAllWindows()
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

        tracklets, success_results = update_trackers(trackers, frame)
        
        front_distance, marker_upper_front, marker_lower_front = compute_distance(tracklets[1], tracklets[3])
        annotate_frame(frame, marker_upper_front, marker_lower_front, front_distance, (255, 0, 0))
        front_distances.append(front_distance)
        front_distances_frame_ids.append(frame_id)

        back_distance, marker_upper_back, marker_lower_back = compute_distance(tracklets[0], tracklets[2])
        annotate_frame(frame, marker_upper_back, marker_lower_back, back_distance, (0, 0, 255))
        back_distances.append(back_distance)
        back_distances_frame_ids.append(frame_id)

        cv2.putText(frame, f"Frame ID {frame_id}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        writer.write(frame)
        cv2.imshow("Preview", frame)
        cv2.waitKey(1)
        
        frame_id += 1

    writer.release()
    save_results(result_filepath, front_distances_frame_ids, front_distances, back_distances)
    plot_results(front_distances_frame_ids, front_distances, back_distances, output_plot_path)

# Initialize Trackers
def initialize_trackers(frame, bboxes):
    trackers = []
    for bbox in bboxes:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        trackers.append(tracker)
    return trackers

# Update Trackers
def update_trackers(trackers, frame):
    tracklets = []
    success_results = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        tracklets.append(Tracklet(bbox))
        success_results.append(success)
    return tracklets, success_results

# Compute Distance
def compute_distance(marker_upper, marker_lower):
    distance = euclidean_distance(marker_upper, marker_lower)
    return distance, marker_upper, marker_lower

# Annotate Frame
def annotate_frame(frame, marker_upper, marker_lower, distance, color):
    cv2.putText(frame, str(np.round(distance, 1)), (marker_upper.center[0] + 50, marker_upper.center[1] + 50), 0, 1.2, (0, 0, 0), 3)
    cv2.line(frame, marker_upper.center, marker_lower.center, color, 6)

# Save Results
def save_results(filepath, frame_ids, front_distances, back_distances):
    data = {
        "frame_ids": frame_ids,
        "front": front_distances,
        "back": back_distances
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

# Plot Results
def plot_results(frame_ids, front_distances, back_distances, output_path):
    front_distances = np.array(front_distances) - front_distances[0]
    back_distances = np.array(back_distances) - back_distances[0]

    plt.figure(figsize=(16, 9))
    plt.plot(frame_ids, front_distances, color="blue")
    plt.plot(frame_ids, back_distances, color="red")
    plt.title("Tracklet distance progression FRONT and BACK")
    plt.xlabel("# Frame")
    plt.ylabel("Euclidean distance relative to starting distance in px")
    plt.legend(["front", "back"])
    plt.savefig(output_path)

if __name__ == "__main__":
    args = parse_arguments()
    process_video(args)
