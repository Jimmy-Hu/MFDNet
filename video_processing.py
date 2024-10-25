import os
import argparse
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
import ffmpeg
import cv2
import numpy as np

def process_video_frame_by_frame(input_file, output_file, model_restoration):
    """
