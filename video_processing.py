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
            ffmpeg
            .input(input_file)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

        # Output
        process2 = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # Process frame (deraining processing)
        while in_bytes := process1.stdout.read(width * height * 3):
            with torch.no_grad(): 
                in_frame = torch.frombuffer(in_bytes, dtype=torch.uint8).float().reshape((1, 3, width, height))
                in_frame_gpu = torch.div(in_frame, 255).to(device='cuda')
                    
                restored = model_restoration(in_frame_gpu)
                restored = torch.clamp(restored[1], 0, 1)
                    
                out_frame = (restored.cpu() * 255).byte().numpy()
            
                # Clear cache and del intermediate vars
                torch.cuda.empty_cache()
                del in_frame_gpu
                del restored

                process2.stdin.write(out_frame.tobytes())

        # Close streams
        process1.stdout.close()
        process2.stdin.close()
        process1.wait()
        process2.wait()

    except ffmpeg.Error as e:
