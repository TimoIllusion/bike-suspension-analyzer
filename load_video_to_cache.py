import os
import argparse

parser = argparse.ArgumentParser(description='Extract images from video')
parser.add_argument('video_path', type=str, help='Path to video file')
parser.add_argument('--output_dir', type=str, default="./.cache/", help='Output directory to save images')
args = parser.parse_args()

video_name = os.path.basename(args.video_path)
images_dir = os.path.join(args.output_dir, video_name)
os.makedirs(images_dir, exist_ok=True)

cmd = f"ffmpeg -i {args.video_path} {images_dir}/%05d.jpg"
print(cmd)
os.system(cmd)