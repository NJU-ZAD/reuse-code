import os

import ffmpeg
from PIL import Image

#! pip3 install ffmpeg-python pillow -y

username = os.getlogin()
if 'SSH_CLIENT' in os.environ or 'SSH_CONNECTION' in os.environ:
    DATA_DISK_DIR = '/data/'+username
else:
    MSI = '/media/'+username+'/PortableSSD'
    if os.path.isdir(MSI) and os.listdir(MSI):
        DATA_DISK_DIR = MSI
    else:
        DATA_DISK_DIR = '/media/'+username+'/数据硬盘'
video_path = os.path.expanduser(
    DATA_DISK_DIR+"/kinetics-dataset/k400/k400_320p/ZZoxcS-rDGE_000288_000298.mp4")

output_image_path = "frame001.jpg"
if not os.path.exists(video_path):
    print(f"Error: '{video_path}' file not found.")
    exit(1)
try:
    (
        ffmpeg
        .input(video_path)
        .output("frame%03d.jpg", vframes=1)
        .run()
    )
    image = Image.open(output_image_path)
    image.show()
finally:
    if os.path.exists(output_image_path):
        os.remove(output_image_path)
