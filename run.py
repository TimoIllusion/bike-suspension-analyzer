import os
import math
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
import pandas
import tkinter as tk
from tkinter import filedialog, messagebox
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, ImageTk

class Constants:
    TRACKER_ROI_CACHE_FILENAME = "_roi_cache.json"
    OUTPUT_RESULT_DATA_JSON_FILE = "results.json"
    OUTPUT_PLOT_FILENAME = "plot.png"
    OUTPUT_VIDEO_FILENAME = "out.mp4"
    OUTPUT_VIDEO_FPS = 30

class FilePathManager:
    def __init__(self, video_path, output_dir, cache_dir):
        self.video_source_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_source_path))[0]
        self.cache_dir = cache_dir
        
        self.tracker_roi_cache_path = os.path.join(self.cache_dir, self.video_name, Constants.TRACKER_ROI_CACHE_FILENAME)
        
        self.out_dir_video_results = os.path.join(output_dir, self.video_name)
        self.result_filepath = os.path.join(self.out_dir_video_results, Constants.OUTPUT_RESULT_DATA_JSON_FILE)
        self.output_video_path = os.path.join(self.out_dir_video_results, Constants.OUTPUT_VIDEO_FILENAME)
        self.output_plot_path = os.path.join(self.out_dir_video_results, Constants.OUTPUT_PLOT_FILENAME)
        
        self.create_directories()
    
    def create_directories(self):
        os.makedirs(self.out_dir_video_results, exist_ok=True)

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
    PLOT_RECT = (400, 420, 480, 250)

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

    def draw_distance_progress(self, frame, frame_ids, front_distances, back_distances, start_frame, end_frame):
        if not frame_ids:
            return

        start_frame = start_frame if start_frame is not None else frame_ids[0]
        end_frame = end_frame if end_frame is not None else frame_ids[-1]

        x, y, width, height = self.PLOT_RECT
        padding_left = 55
        padding_right = 20
        padding_top = 32
        padding_bottom = 42
        plot_x = x + padding_left
        plot_y = y + padding_top
        plot_width = width - padding_left - padding_right
        plot_height = height - padding_top - padding_bottom

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (245, 245, 245), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), 1)

        front_relative = np.array(front_distances) - front_distances[0]
        back_relative = np.array(back_distances) - back_distances[0]
        y_values = np.concatenate([front_relative, back_relative, np.array([0])])
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_span = y_max - y_min
        if y_span < 1:
            y_min -= 1
            y_max += 1
        else:
            y_padding = y_span * 0.15
            y_min -= y_padding
            y_max += y_padding

        def map_x(frame_id):
            frame_span = max(end_frame - start_frame, 1)
            progress = (frame_id - start_frame) / frame_span
            progress = min(max(progress, 0), 1)
            return int(plot_x + progress * plot_width)

        def map_y(value):
            value_span = max(y_max - y_min, 1)
            progress = (value - y_min) / value_span
            return int(plot_y + plot_height - progress * plot_height)

        def draw_polyline(values, color):
            points = [
                (map_x(frame_id), map_y(value))
                for frame_id, value in zip(frame_ids, values)
            ]
            if len(points) == 1:
                cv2.circle(frame, points[0], 3, color, -1)
            else:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

        cv2.line(frame, (plot_x, plot_y), (plot_x, plot_y + plot_height), (80, 80, 80), 1)
        cv2.line(
            frame,
            (plot_x, plot_y + plot_height),
            (plot_x + plot_width, plot_y + plot_height),
            (80, 80, 80),
            1
        )

        zero_y = map_y(0)
        if plot_y <= zero_y <= plot_y + plot_height:
            cv2.line(frame, (plot_x, zero_y), (plot_x + plot_width, zero_y), (180, 180, 180), 1, cv2.LINE_AA)

        draw_polyline(front_relative, (255, 0, 0))
        draw_polyline(back_relative, (0, 0, 255))

        current_x = map_x(frame_ids[-1])
        cv2.line(frame, (current_x, plot_y), (current_x, plot_y + plot_height), (70, 70, 70), 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "Distance progress",
            (x + 16, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "front",
            (plot_x + 8, y + height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "back",
            (plot_x + 72, y + height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"{y_max:.0f}",
            (x + 10, plot_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (40, 40, 40),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"{y_min:.0f}",
            (x + 10, plot_y + plot_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (40, 40, 40),
            1,
            cv2.LINE_AA
        )

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
    def __init__(self, roi_file_path):
        self.roi_file_path = roi_file_path

    def cached_roi_location_available(self):
        return os.path.exists(self.roi_file_path)

    def load_cached_roi_location(self):
        logger.warning(f"Loading cached tracklet locations from {self.roi_file_path}")
        with open(self.roi_file_path, "r") as f:
            bboxes = json.load(f)
        return bboxes["bbox_top_front"], bboxes["bbox_bottom_front"], bboxes["bbox_top_back"], bboxes["bbox_bottom_back"]

    def manually_set_roi_locations(self, frame):
        bbox_top_front = cv2.selectROI('top front marker', frame, True)
        bbox_bottom_front = cv2.selectROI('bottom front marker', frame, True)
        bbox_top_back = cv2.selectROI('top back marker', frame, True)
        bbox_bottom_back = cv2.selectROI('bottom back marker', frame, True)

        bboxes = {
            "bbox_top_front": bbox_top_front,
            "bbox_bottom_front": bbox_bottom_front,
            "bbox_top_back": bbox_top_back,
            "bbox_bottom_back": bbox_bottom_back
        }

        roi_file_dir = os.path.dirname(self.roi_file_path)
        os.makedirs(roi_file_dir, exist_ok=True)

        with open(self.roi_file_path, "w") as f:
            json.dump(bboxes, f, indent=4)

        return bbox_top_front, bbox_bottom_front, bbox_top_back, bbox_bottom_back

    def try_to_load_roi_cache_or_set_manually(self, frame):
        if self.cached_roi_location_available():
            return self.load_cached_roi_location()
        else:
            return self.manually_set_roi_locations(frame)

    def reset_roi(self):
        if self.cached_roi_location_available():
            os.remove(self.roi_file_path)
            logger.info(f"Deleted cached ROI file: {self.roi_file_path}")
        else:
            logger.warning("No cached ROI file found to delete.")

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
        
        df = pandas.read_json(filepath)
        filepath_excel = filepath.replace(".json", ".xlsx")
        df.to_excel(filepath_excel)

class VideoProcessor:
    def __init__(self, video_path, output_dir, cache_dir, start, end):
        self.file_manager = FilePathManager(video_path, output_dir, cache_dir)
        self.video = VideoFileClip(video_path)
        self.start_frame = start
        self.end_frame = end if end is not None else int(self.video.fps * self.video.duration) - 1
        self.front_distances = []
        self.front_distances_frame_ids = []
        self.back_distances = []
        self.back_distances_frame_ids = []
        self.initialized = False

    def process_video(self):
        self.initialize_processing_tools()
        
        logger.info("Processing started")

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
            
        logger.info("Processing finished")
        self.finalize_processing()

    def initialize_processing_tools(self):
        self.visualizer = Visualizer(self.file_manager.output_video_path, Constants.OUTPUT_VIDEO_FPS, (1280, 720))
        self.roi_manager = ROIManager(self.file_manager.tracker_roi_cache_path)

    def load_and_prepare_frame(self, frame_id):
        logger.trace(f"Processing frame {frame_id}")

        frame = self.video.get_frame(frame_id / self.video.fps)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.resize(frame, (1280, 720))

    def initialize_trackers(self, frame):
        bbox_top_front, bbox_bottom_front, bbox_top_back, bbox_bottom_back = self.roi_manager.try_to_load_roi_cache_or_set_manually(frame)
        self.tracker = TrackerManager(frame, [bbox_top_front, bbox_bottom_front, bbox_top_back, bbox_bottom_back])
        self.initialized = True
        cv2.destroyAllWindows()
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    def update_trackers(self, frame):
        return self.tracker.update_trackers(frame)

    def process_frame(self, frame, frame_id, tracklets):
        front_distance, marker_upper_front, marker_lower_front = self.compute_distance(tracklets[0], tracklets[1])

        self.visualizer.annotate_frame(frame, marker_upper_front, marker_lower_front, front_distance, (255, 0, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[0], (0, 255, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[1], (0, 255, 0))

        self.front_distances.append(front_distance)
        self.front_distances_frame_ids.append(frame_id)

        back_distance, marker_upper_back, marker_lower_back = self.compute_distance(tracklets[2], tracklets[3])

        self.visualizer.annotate_frame(frame, marker_upper_back, marker_lower_back, back_distance, (0, 0, 255))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[2], (0, 255, 0))
        self.visualizer.draw_marker_as_rectangle(frame, tracklets[3], (0, 255, 0))
        
        self.back_distances.append(back_distance)
        self.back_distances_frame_ids.append(frame_id)

        cv2.putText(frame, f"Frame ID {frame_id}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        self.visualizer.draw_distance_progress(
            frame,
            self.front_distances_frame_ids,
            self.front_distances,
            self.back_distances,
            self.start_frame,
            self.end_frame
        )
        self.visualizer.write_frame(frame)
        cv2.imshow("Preview", frame)

    def check_user_interrupt(self):
        return cv2.waitKey(1) & 0xFF in [ord('q'), 27]

    def finalize_processing(self):
        self.visualizer.release()
        ResultSaver.save_results(self.file_manager.result_filepath, self.front_distances_frame_ids, self.front_distances, self.back_distances)
        Visualizer.plot_results(self.front_distances_frame_ids, self.front_distances, self.back_distances, self.file_manager.output_plot_path)
        logger.info(f"Results saved to {self.file_manager.out_dir_video_results}")

    @staticmethod
    def compute_distance(marker_upper, marker_lower):
        distance = euclidean_distance(marker_upper, marker_lower)
        return distance, marker_upper, marker_lower

class BikeSuspensionAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bike Suspension Analyzer")
        self.geometry("1000x700")

        self.video_path = ""
        self.output_dir = "./output/"
        self.cache_dir = "./.cache/"
        self.start_frame = 1
        self.end_frame = None

        self.video = None  # VideoFileClip object

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Video Path").grid(row=0, column=0, padx=10, pady=10)
        self.video_entry = tk.Entry(self, width=40)
        self.video_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self, text="Output Directory").grid(row=1, column=0, padx=10, pady=10)
        self.output_entry = tk.Entry(self, width=40)
        self.output_entry.insert(0, self.output_dir)
        self.output_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self, text="Cache Directory").grid(row=2, column=0, padx=10, pady=10)
        self.cache_entry = tk.Entry(self, width=40)
        self.cache_entry.insert(0, self.cache_dir)
        self.cache_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self, text="Start Frame").grid(row=3, column=0, padx=10, pady=10)
        self.start_entry = tk.Entry(self, width=40)
        self.start_entry.insert(0, self.start_frame)
        self.start_entry.grid(row=3, column=1, padx=10, pady=10)
        self.start_entry.bind('<Return>', self.update_start_frame)

        tk.Label(self, text="End Frame").grid(row=4, column=0, padx=10, pady=10)
        self.end_entry = tk.Entry(self, width=40)
        self.end_entry.grid(row=4, column=1, padx=10, pady=10)
        self.end_entry.bind('<Return>', self.update_end_frame)

        tk.Button(self, text="Start Analysis", command=self.start_analysis).grid(row=5, column=0, columnspan=3, pady=10)
        tk.Button(self, text="Reset ROI", command=self.reset_roi).grid(row=6, column=0, columnspan=3, pady=10)

        self.frame_label = tk.Label(self)
        self.frame_label.grid(row=7, column=0, columnspan=3, pady=10)

        self.start_frame_label = tk.Label(self)
        self.start_frame_label.grid(row=8, column=0, padx=10, pady=10)

        self.end_frame_label = tk.Label(self)
        self.end_frame_label.grid(row=8, column=1, padx=10, pady=10)
        # add text above the image
        tk.Label(self, text="Start Frame").grid(row=9, column=0, padx=10, pady=10)
        tk.Label(self, text="End Frame").grid(row=9, column=1, padx=10, pady=10)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path = file_path
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, self.video_path)
            self.video = VideoFileClip(self.video_path)
            self.end_frame = int(self.video.fps * self.video.duration) - 2 # workaround for the last frame not being displayed
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, self.end_frame)
            self.update_start_frame()
            self.update_end_frame()

    def start_analysis(self):
        self.video_path = self.video_entry.get()
        self.output_dir = self.output_entry.get()
        self.cache_dir = self.cache_entry.get()
        self.start_frame = int(self.start_entry.get())
        self.end_frame = int(self.end_entry.get()) if self.end_entry.get() else None

        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        args = argparse.Namespace(
            video_path=self.video_path,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            start=self.start_frame,
            end=self.end_frame
        )

        processor = VideoProcessor(args.video_path, args.output_dir, args.cache_dir, args.start, args.end)
        processor.process_video()

    def reset_roi(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        cache_dir = self.cache_entry.get()
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        roi_file_path = os.path.join(cache_dir, video_name, Constants.TRACKER_ROI_CACHE_FILENAME)
        
        roi_manager = ROIManager(roi_file_path)
        roi_manager.reset_roi()
        messagebox.showinfo("Reset ROI", "The ROI cache has been reset. You will be prompted to set the ROI again during the next analysis.")

    def display_frame(self, frame_number, label, scale=200):
        if self.video:
            frame = self.video.get_frame(frame_number / self.video.fps)
            frame = cv2.resize(frame, (scale, scale))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)

    def update_start_frame(self, event=None):
        try:
            frame_number = int(self.start_entry.get())
            self.display_frame(frame_number, self.start_frame_label, 200)
        except ValueError:
            pass

    def update_end_frame(self, event=None):
        try:
            frame_number = int(self.end_entry.get())
            self.display_frame(frame_number, self.end_frame_label, 200)
        except ValueError:
            pass

if __name__ == "__main__":
    app = BikeSuspensionAnalyzerApp()
    app.mainloop()
