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
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

    parser.add_argument('--weights', default='./checkpoints/checkpoints_mfd.pth', type=str,
                        help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model_restoration = mfdnet()
    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.eval().cuda()
    
    input_video = "Input_Video.mp4"
    output_video = "Output_Video.mp4"

    process_video_frame_by_frame(input_video, output_video, model_restoration)

    end_time = time.time()
    print(end_time - start_time)