import os
import argparse

parser = argparse.ArgumentParser(description='Extract images from video')
parser.add_argument('video_path', type=str, help='Path to video file')
args = parser.parse_args()

images_dir = os.path.dirname(args.video_path) + "/" + os.path.splitext(os.path.basename(args.video_path))[0]
os.makedirs(images_dir, exist_ok=True)

cmd = f"ffmpeg -i {args.video_path} {images_dir}/%05d.jpg"
print(cmd)
os.system(cmd)