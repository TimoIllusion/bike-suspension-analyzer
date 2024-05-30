import os
import math
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

class Constants:
    CACHE_FILENAME = "_roi_cache.json"
    RESULT_FILENAME = "results.json"
    OUTPUT_VIDEO_FILENAME = "out.mp4"
    OUTPUT_PLOT_FILENAME = "plot.png"
    FPS = 30

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze motorcycle suspension')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_dir', type=str, default="./output/", help='Output directory to save results')
    parser.add_argument('--cache_dir', type=str, default="./.cache/", help='Cache directory to save images')
    parser.add_argument('--start', type=int, default=1, help='Start frame number (has to start with 1) (default: 1)')
    parser.add_argument('--end', type=int, default=None, help='End frame number (default: None)')
    args = parser.parse_args()

    assert args.start > 0, "Start frame has to be greater than 0 due to ffmpeg generated frames starting with 00001"

    return args

class FilePathManager:
    def __init__(self, video_path, output_dir, cache_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.images_dir = os.path.join(self.cache_dir, self.video_name)
        self.out_dir_with_video_name = os.path.join(self.output_dir, self.video_name)
        self.cache_path = self.get_cache_path(self.images_dir)
        self.result_filepath = os.path.join(self.out_dir_with_video_name, Constants.RESULT_FILENAME)
        self.output_video_path = os.path.join(self.out_dir_with_video_name, Constants.OUTPUT_VIDEO_FILENAME)
        self.output_plot_path = os.path.join(self.out_dir_with_video_name, Constants.OUTPUT_PLOT_FILENAME)
        os.makedirs(self.out_dir_with_video_name, exist_ok=True)

    @staticmethod
    def get_cache_path(images_dir):
        return os.path.join(images_dir, Constants.CACHE_FILENAME)

class VideoUtils:
    @staticmethod
    def extract_images_if_not_existing(video_path, images_dir):
        if os.path.exists(images_dir):
            logger.warning(f"Images directory {images_dir} already exists. Skipping extraction.")
            return
        os.makedirs(images_dir, exist_ok=True)
        cmd = f"ffmpeg -i {video_path} {images_dir}/%05d.jpg"
        logger.info(f"Executing command: {cmd}")
        os.system(cmd)

    @staticmethod
    def get_image_count(images_dir):
        return len([f for f in os.listdir(images_dir) if f.endswith('.jpg') and f != Constants.CACHE_FILENAME])

class Tracklet:
    def __init__(self, bbox):
        x, y, w, h = map(int, bbox)
        self.center = (int(x + w / 2), int(y + h / 2))
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def euclidean_distance(a: Tracklet, b: Tracklet) -> float:
    return math.sqrt((b.center[0] - a.center[0]) ** 2 + (b.center[1] - a.center[1]) ** 2)

class TrackerManager:
    def __init__(self, frame, bboxes):
        self.trackers = self.initialize_trackers(frame, bboxes)

    @staticmethod
    def initialize_trackers(frame, bboxes):
        trackers = []
        for bbox in bboxes:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            trackers.append(tracker)
        return trackers

    def update_trackers(self, frame):
        tracklets = []
        success_results = []
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            tracklets.append(Tracklet(bbox))
            success_results.append(success)
        return tracklets, success_results

class Visualizer:
    def __init__(self, output_video_path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    def annotate_frame(self, frame, marker_upper, marker_lower, distance, color):
        cv2.putText(frame, str(np.round(distance, 1)), (marker_upper.center[0] + 50, marker_upper.center[1] + 50), 0, 1.2, (0, 0, 0), 3)
        cv2.line(frame, marker_upper.center, marker_lower.center, color, 6)

    def draw_marker_as_rectangle(self, frame, marker, color):
        cv2.rectangle(frame, (marker.x, marker.y), (marker.x + marker.w, marker.y + marker.h), color, 2)

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    @staticmethod
    def plot_results(frame_ids, front_distances, back_distances, output_plot_path):
        front_distances = np.array(front_distances) - front_distances[0]
        back_distances = np.array(back_distances) - back_distances[0]

        plt.figure(figsize=(16, 9))
        plt.plot(frame_ids, front_distances, color="blue")
        plt.plot(frame_ids, back_distances, color="red")
        plt.title("Tracklet distance progression FRONT and BACK")
        plt.xlabel("# Frame")
        plt.ylabel("Euclidean distance relative to starting distance in px")
        plt.legend(["front", "back"])
        plt.savefig(output_plot_path)

class ROIManager:
    def __init__(self, cache_path):
        self.cache_path = cache_path

    def cached_roi_location_available(self):
        return os.path.exists(self.cache_path)

    def load_cached_roi_location(self):
        logger.warning(f"Loading cached tracklet locations from {self.cache_path}")
        with open(self.cache_path, "r") as f:
            bboxes = json.load(f)
        return bboxes["bbox_top_left"], bboxes["bbox_top_right"], bboxes["bbox_bottom_left"], bboxes["bbox_bottom_right"]

    def manually_set_roi_locations(self, frame):
        bbox_top_left = cv2.selectROI('bbox_top_left', frame, True)
        bbox_bottom_left = cv2.selectROI('bbox_bottom_left', frame, True)
        bbox_top_right = cv2.selectROI('bbox_top_right', frame, True)
        bbox_bottom_right = cv2.selectROI('bbox_bottom_right', frame, True)

        bboxes = {
            "bbox_top_left": bbox_top_left, 
            "bbox_top_right": bbox_top_right, 
            "bbox_bottom_left": bbox_bottom_left, 
            "bbox_bottom_right": bbox_bottom_right
        }

        cache_parent_dir = os.path.dirname(self.cache_path)
        os.makedirs(cache_parent_dir, exist_ok=True)

        with open(self.cache_path, "w") as f:
            json.dump(bboxes, f, indent=4)

        return bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right

    def try_to_load_roi_cache_or_set_manually(self, frame):
        if self.cached_roi_location_available():
            return self.load_cached_roi_location()
        else:
            return self.manually_set_roi_locations(frame)

class ResultSaver:
    @staticmethod
    def save_results(filepath, frame_ids, front_distances, back_distances):
        data = {
            "frame_ids": frame_ids,
            "front": front_distances,
            "back": back_distances
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

class VideoProcessor:
    def __init__(self, args):
        self.file_manager = FilePathManager(args.video_path, args.output_dir, args.cache_dir)
        self.start_frame = args.start
        self.end_frame = args.end
        self.front_distances = []
        self.front_distances_frame_ids = []
        self.back_distances = []
        self.back_distances_frame_ids = []
        self.initialized = False

    def process_video(self):
        self.extract_and_prepare_images()
        self.set_end_frame_if_none()
        self.initialize_processing_tools()

        for frame_id in tqdm(range(self.start_frame, self.end_frame + 1)):
            frame = self.load_and_prepare_frame(frame_id)
            if frame is None:
                continue

            if not self.initialized:
                self.initialize_trackers(frame)
            
            tracklets, success_results = self.update_trackers(frame)
            self.process_frame(frame, frame_id, tracklets)
            if self.check_user_interrupt():
                break

        self.finalize_processing()

    def extract_and_prepare_images(self):
        VideoUtils.extract_images_if_not_existing(self.file_manager.video_path, self.file_manager.images_dir)

    def set_end_frame_if_none(self):
        if self.end_frame is None:
            self.end_frame = VideoUtils.get_image_count(self.file_manager.images_dir)
            logger.info(f"End frame not specified. Setting to {self.end_frame}")

    def initialize_processing_tools(self):
        self.visualizer = Visualizer(self.file_manager.output_video_path, Constants.FPS, (1280, 720))
        self.roi_manager = ROIManager(self.file_manager.cache_path)
        os.makedirs(self.file_manager.out_dir_with_video_name, exist_ok=True)

    def load_and_prepare_frame(self, frame_id):
        frame_file = os.path.join(self.file_manager.images_dir, f"{frame_id:05d}.jpg")
        logger.trace(f"Processing frame {frame_id} from {frame_file}")

        if not os.path.exists(frame_file):
            logger.error(f"Frame {frame_file} not found. Skipping frame.")
            return None

        frame = cv2.imread(frame_file)
        return cv2.resize(frame, (1280, 720))

    def initialize_trackers(self, frame):
        bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right = self.roi_manager.try_to_load_roi_cache_or_set_manually(frame)
        self.tracker = TrackerManager(frame, [bbox_top_left, bbox_top_right, bbox_bottom_left, bbox_bottom_right])
        self.initialized = True
        cv2.destroyAllWindows()
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    def update_trackers(self, frame):
        return self.tracker.update_trackers(frame)

    def process_frame(self, frame, frame_id, tracklets):
        front_distance, marker_upper_front, marker_lower_front = self.compute_distance(tracklets[1], tracklets[3])
        self.visualizer.annotate_frame(frame, marker_upper_front, marker_lower_front, front_distance, (255, 0, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[1], (0, 255, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[3], (0, 255, 0))
        self.front_distances.append(front_distance)
        self.front_distances_frame_ids.append(frame_id)

        back_distance, marker_upper_back, marker_lower_back = self.compute_distance(tracklets[0], tracklets[2])
        self.visualizer.annotate_frame(frame, marker_upper_back, marker_lower_back, back_distance, (0, 0, 255))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[0], (0, 255, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[2], (0, 255, 0))
        self.back_distances.append(back_distance)
        self.back_distances_frame_ids.append(frame_id)

        cv2.putText(frame, f"Frame ID {frame_id}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        self.visualizer.write_frame(frame)
        cv2.imshow("Preview", frame)

    def check_user_interrupt(self):
        return cv2.waitKey(1) & 0xFF in [ord('q'), 27]

    def finalize_processing(self):
        self.visualizer.release()
        ResultSaver.save_results(self.file_manager.result_filepath, self.front_distances_frame_ids, self.front_distances, self.back_distances)
        Visualizer.plot_results(self.front_distances_frame_ids, self.front_distances, self.back_distances, self.file_manager.output_plot_path)

    @staticmethod
    def compute_distance(marker_upper, marker_lower):
        distance = euclidean_distance(marker_upper, marker_lower)
        return distance, marker_upper, marker_lower

if __name__ == "__main__":
    args = parse_arguments()
    processor = VideoProcessor(args)
    processor.process_video()
