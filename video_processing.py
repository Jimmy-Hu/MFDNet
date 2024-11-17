import argparse
import os
import time

import cv2
import ffmpeg
import numpy as np
import torch
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_RGB import get_test_data
from MFDNet import HPCNet as mfdnet


def process_video_frame_by_frame(input_file, output_file, model_restoration):
    """
    Decodes a video frame by frame, processes each frame,
    and re-encodes to a new video.

    Args:
        input_file: Path to the input video file.
        output_file: Path to the output video file.
    """
    try:
        # Probe for video information
        probe = ffmpeg.probe(input_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Input
        process1 = (
