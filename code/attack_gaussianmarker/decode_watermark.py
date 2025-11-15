import os
import torch
import torch.nn as nn
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, modified_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import random
import copy
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from hidden.hidden_images import encoder, decoder, params, str2msg, msg2str, default_transform, NORMALIZE_IMAGENET, UNNORMALIZE_IMAGENET
from einops import reduce, repeat, rearrange
from utils.system_utils import searchForMaxIteration

import seaborn as sns
import matplotlib.pyplot as plt

import copy
from torchvision.transforms.v2 import JPEG
from models.pointnet_cls import get_model, get_loss
from utils.aug_utils import addNoise, Resize
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

random_seed = 0  # 你可以选择任何整数作为随机种子
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def calculate_all_metrics(decoded_msg, true_msg):
    """
    Calculates BAR, WUS, and IDS from the decoded and true messages.

    Args:
        decoded_msg (torch.Tensor): The boolean tensor of the message extracted from the model.
        true_msg (torch.Tensor): The boolean tensor of the ground truth message.

    Returns:
        tuple: A tuple containing (bar, wus, ids).
    """
    # Ensure tensors are boolean and on the same device for comparison
    decoded_msg = decoded_msg.bool()
    true_msg = true_msg.bool()

    num_bits = true_msg.numel()

    # 1. Calculate Bit Accuracy Rate (BAR)
    bar = torch.sum(decoded_msg == true_msg) / num_bits

    # 2. Calculate Watermark Uncertainty Score (WUS)
    wus = 1.0 - 2.0 * torch.abs(bar - 0.5)

    # 3. Calculate Information Destruction Score (IDS) via MCC
    # Get confusion matrix elements
    tp = torch.sum((decoded_msg == 1) & (true_msg == 1)).float()
    tn = torch.sum((decoded_msg == 0) & (true_msg == 0)).float()
    fp = torch.sum((decoded_msg == 1) & (true_msg == 0)).float()
    fn = torch.sum((decoded_msg == 0) & (true_msg == 1)).float()

    # Calculate Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Add a small epsilon to the denominator to prevent division by zero
    mcc = numerator / (denominator + 1e-8)

    # Calculate IDS
    ids = 1.0 - torch.abs(mcc)

    return bar.item(), wus.item(), ids.item()

def load_model(model_path):
    # Load the GaussianModel from the specified PLY file
    gaussians = GaussianModel(sh_degree=3)
    ply_file = model_path + iteration_path
    gaussians.load_ply(ply_file)
    return gaussians

def extract_watermark(dataset, opt, pipe, model_path, args):
    # Load the trained GaussianModel
    gaussians = load_model(model_path)

    if "blender" in args.wm_source_path:
        dataset.resolution = 4
    elif "LLFF" in args.wm_source_path:
        dataset.resolution = 2

    # Create a Scene object
    dataset.model_path, dataset.source_path = args.wm_model_path, args.wm_source_path
    scene = Scene(dataset, gaussians)

    # Set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get the test cameras
    test_cameras = scene.getTestCameras()
    # test_cameras = scene.getTrainCameras()

    # Initialize the decoder
    decoder_model = decoder

    # Create a directory to save the results
    # results_dir = os.path.join(model_path, f"watermark_extraction_results")
    # os.makedirs(results_dir, exist_ok=True)

    # Create the true message
    true_msg = create_message(args.input_msg)

    # Variables to track overall accuracy
    total_bit_accuracy = 0.0
    total_bar = 0.0
    total_wus = 0.0
    total_ids = 0.0
    num_cameras = len(test_cameras)

    # Extract watermark from each test camera
    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Extracting watermark from test cameras")):
        # Render the image
        render_pkg = render(viewpoint, gaussians, pipe, background)
        image = render_pkg["render"]

        # Normalize the image
        nom_image = NORMALIZE_IMAGENET(image.unsqueeze(0))

        # Decode the watermark
        with torch.no_grad():
            ft = decoder_model(nom_image)

        # Convert the decoded message to a binary string
        decoded_msg = ft > 0
        decoded_str = msg2str(decoded_msg.squeeze(0).cpu().numpy())

        bar, wus, ids = calculate_all_metrics(decoded_msg, true_msg)
        total_bar += bar
        total_wus += wus
        total_ids += ids

        # Calculate bit accuracy for this camera
        bit_accuracy = torch.sum(~torch.logical_xor(decoded_msg, true_msg)) / params.num_bits

        # Add to total accuracy
        total_bit_accuracy += bit_accuracy.item()

        # Save the results
        result = {
            "rendered_image": image,
            "decoded_message": decoded_str,
            "bit_accuracy": bit_accuracy.item()
        }

        # Save the rendered image and the decoded message
        # torchvision.utils.save_image(image, os.path.join(results_dir, f"rendered_image_{idx}.png"))
        # with open(os.path.join(results_dir, f"decoded_message_{idx}.txt"), "w") as f:
        #     f.write(decoded_str)

        print(f"Extracted watermark for camera {idx}: {decoded_str}, Bit Accuracy: {bit_accuracy.item()}")
        print(f"  -> Metrics: BAR={bar:.4f}, WUS={wus:.4f}, IDS={ids:.4f}")

    # Calculate and print overall bit accuracy
    overall_bit_accuracy = total_bit_accuracy / num_cameras
    overall_bar = total_bar / num_cameras
    overall_wus = total_wus / num_cameras
    overall_ids = total_ids / num_cameras
    print(f"Overall Bit Accuracy across all {num_cameras} cameras: {overall_bit_accuracy}")
    print("\n" + "=" * 20 + " Overall Results " + "=" * 20)
    print(f"Overall Bit Accuracy (BAR) across {num_cameras} cameras: {overall_bar:.4f}")
    print(f"Overall Watermark Uncertainty (WUS) across {num_cameras} cameras: {overall_wus:.4f}")
    print(f"Overall Information Destruction (IDS) across {num_cameras} cameras: {overall_ids:.4f}")
    print("=" * 57)

    # Save overall accuracy to a file
    # with open(os.path.join(results_dir, "overall_bit_accuracy.txt"), "w") as f:
    #     f.write(f"Overall Bit Accuracy: {overall_bit_accuracy}")

def create_message(input_msg, random_msg=False):
    # Create message
    if random_msg:
        msg_ori = torch.randint(0, 2, (1, params.num_bits), device="cuda").bool() # b k
    else:
        msg_ori = torch.Tensor(str2msg(input_msg)).unsqueeze(0)
    msg_ori = msg_ori.cuda()
    return msg_ori

if __name__ == "__main__":
    # Set up command line argument parser
    iteration_path = "/point_cloud/iteration_2000/point_cloud.ply"

    parser = ArgumentParser(description="Extract watermark from a trained 3DGS model")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--wm_source_path', default="/sda1/qyzeng/GaussianMarker/data/blender/ficus", type=str, help="Path to the trained model")

    # parser.add_argument('--wm_model_path', default="output/lego_attack/merged_gaussians", type=str, help="Path to the trained model")
    # parser.add_argument('--wm_model_path', default="output/drums", type=str, help="Path to the trained model")
    parser.add_argument('--wm_model_path', default="output/ficus_reconstruct", type=str,  help="Path to the trained model")
    # parser.add_argument('--wm_model_path', default="output/lego_reconstruct", type=str, help="Path to the trained model")
    parser.add_argument('--input_msg', type=str, default="111010110101000001010111010011010100010000100111", help="Input message for verification")  # "111010110101000001010111010011010100010000100111"  "000000000000000000000000000000000000000000000000"

    args = parser.parse_args(sys.argv[1:])

    lp.model_path, lp._model_path = args.wm_model_path, args.wm_model_path


    # Configure and run watermark extraction
    torch.autograd.set_detect_anomaly(False)
    extract_watermark(lp.extract(args), op.extract(args), pp.extract(args), args.wm_model_path, args)

# 记得调整init代码的_resolution，blender=4 LLFF=2！！！！ 记得调整source_path