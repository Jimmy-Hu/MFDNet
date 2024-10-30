import argparse
import os
import time

import cv2
import ffmpeg
import numpy as np
import torch
from skimage import img_as_ubyte

def process_video_frame_by_frame(input_file, output_file, model_restoration):
    """
    Decodes a video frame by frame, processes each frame,
