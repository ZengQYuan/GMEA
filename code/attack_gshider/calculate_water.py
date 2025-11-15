import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# Assuming all custom modules are accessible from the script's location
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, OptimizationParams, PipelineParams
from decoder import SimpleCNN, WatermarkCNN  # Assuming these are in your project
from utils.general_utils import safe_state
from plyfile import PlyData
import numpy as np
import random

# --- New Imports for Metrics Calculation ---
from PIL import Image
import torchvision.transforms.functional as TF
import torchmetrics
import torchvision.transforms as T
# --- Seed and Environment Setup ---
random_seed = 0  # Fixed random seed for reproducibility
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# ------------------------------------------------------------------------------------
#               UNCHANGED FUNCTION FROM YOUR PROVIDED CODE
# ------------------------------------------------------------------------------------

def load_ply_as_initialization(ply_path, dataset, opt):
    """
    Loads Gaussian model parameters from a PLY file.
    This version is specifically designed for models with 'semantic_' features
    instead of traditional color features.
    """
    gaussians = GaussianModel(dataset.sh_degree)

    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"[ERROR] Could not read PLY file at: {ply_path}. Error: {e}")
        sys.exit(1)

    data = plydata.elements[0].data

    # Check for required attributes
    required_attrs = ['x', 'y', 'z', 'opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    for attr in required_attrs:
        if attr not in data.dtype.names:
            raise ValueError(f"ERROR: PLY file '{ply_path}' is missing the required attribute '{attr}'.")

    xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
    opacities = data['opacity'].reshape(-1, 1)
    scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
    rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()

    # Dynamically find and load semantic features
    semantic_feature_names = [name for name in data.dtype.names if name.startswith('semantic_')]
    if not semantic_feature_names:
        raise ValueError(f"ERROR: PLY file '{ply_path}' does not contain any 'semantic_' feature fields.")

    semantic_dim = len(semantic_feature_names)
    print(f"[INFO] Found {semantic_dim}-dimensional semantic features in the PLY file.")

    semantic_features = np.stack([data[name] for name in semantic_feature_names]).transpose()
    semantic_features = semantic_features.reshape(-1, 1, semantic_dim)

    # Assign to GaussianModel as torch tensors
    gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    # Note: The attribute name must match what is used in your custom GaussianModel
    gaussians._semantic_feature = nn.Parameter(
        torch.tensor(semantic_features, dtype=torch.float, device="cuda").requires_grad_(True))

    # Initialize other required tensors for rendering
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    # Call training_setup to initialize optimizers, etc.
    gaussians.training_setup(opt)

    return gaussians


# ------------------------------------------------------------------------------------
#                     NEW FUNCTION FOR METRIC CALCULATION
# ------------------------------------------------------------------------------------
def calculate_watermark_metrics(dataset, opt, pipe, checkpoint_path, ply_path, gt_watermark_path):
    """
    Loads a 3DGS model and a watermark decoder, renders the 2D watermark,
    loads a ground truth watermark, and computes SSIM, PSNR, and MSE between them.
    """
    # --- 1. Initialize Models and Scene ---
    print("\n[INFO] Initializing models and scene for metrics calculation...")
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), shuffle=False)
    waternet = WatermarkCNN().cuda()  # We only need the watermark decoder

    # --- 2. Load Pre-trained Weights ---
    print(f"\n[INFO] Loading Gaussian model from PLY: {ply_path}")
    gaussians = load_ply_as_initialization(ply_path, dataset, opt)

    waternet_checkpoint_path = checkpoint_path.replace('.pth', '_waternet.pth')
    if os.path.exists(waternet_checkpoint_path):
        waternet.load_state_dict(torch.load(waternet_checkpoint_path, map_location="cuda"))
        print(f"[INFO] Successfully loaded Watermark decoder from: {waternet_checkpoint_path}")
    else:
        print(f"[ERROR] Watermark decoder checkpoint not found at: {waternet_checkpoint_path}")
        return

    # **CRITICAL STEP**: Set model to evaluation mode
    waternet.eval()
    print("[INFO] Model set to evaluation mode.")

    # --- 3. Render Watermark and Load Ground Truth ---
    with torch.no_grad():
        # Select a camera viewpoint (e.g., the first training camera)
        viewpoint_cam = scene.getTrainCameras()[0]
        print(f"\n[INFO] Selected camera '{viewpoint_cam.image_name}' for rendering.")

        # A. Render the 3D Gaussians to get the intermediate feature map
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        feature_image = render_pkg["render"]

        # B. Decode the feature map to get the rendered watermark
        rendered_watermark = waternet(feature_image.unsqueeze(0)).squeeze(0)
        print(f"[INFO] Rendered watermark generated. Shape: {rendered_watermark.shape}")

        # C. Load the ground truth watermark image from file
        if not os.path.exists(gt_watermark_path):
            print(f"[ERROR] Ground truth watermark file not found at: {gt_watermark_path}")
            return

        gt_image_pil = Image.open(gt_watermark_path)
        gt_watermark = TF.to_tensor(gt_image_pil).cuda()
        print(f"[INFO] Ground truth watermark loaded. Initial Shape: {gt_watermark.shape}")

        # Ensure channel count matches
        if rendered_watermark.shape[0] != gt_watermark.shape[0]:
            print(
                f"[WARNING] Channel mismatch! Rendered: {rendered_watermark.shape[0]}, GT: {gt_watermark.shape[0]}. Attempting to convert GT to match rendered.")
            mode = "L" if rendered_watermark.shape[0] == 1 else "RGB"
            gt_image_pil = gt_image_pil.convert(mode)
            gt_watermark = TF.to_tensor(gt_image_pil).cuda()
            print(f"[INFO] Converted GT watermark channel count. Shape: {gt_watermark.shape}")

        # <<< FIX: Resize ground truth watermark to match rendered watermark >>>
        target_size = (rendered_watermark.shape[1], rendered_watermark.shape[2])  # (H, W)
        gt_watermark = TF.resize(gt_watermark, size=target_size)
        print(f"[INFO] Resized ground truth watermark to match rendered shape. Final Shape: {gt_watermark.shape}")

    # --- 4. Calculate and Display Metrics ---
    print("\n[INFO] Calculating metrics...")

    # Add a batch dimension for metric functions (N, C, H, W)
    rendered_batch = rendered_watermark.unsqueeze(0)
    gt_batch = gt_watermark.unsqueeze(0)

    # Ensure tensors are float and in [0, 1] range for metrics
    rendered_batch = rendered_batch.float()
    gt_batch = gt_batch.float()

    plt.imshow(rendered_batch[0].cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()

    # Calculate SSIM (Structural Similarity Index Measure)
    ssim_val = torchmetrics.functional.structural_similarity_index_measure(
        preds=rendered_batch, target=gt_batch, data_range=1.0
    )

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    psnr_val = torchmetrics.functional.peak_signal_noise_ratio(
        preds=rendered_batch, target=gt_batch, data_range=1.0
    )

    # Calculate MSE (Mean Squared Error)
    mse_val = torch.nn.functional.mse_loss(rendered_batch, gt_batch)

    print("\n--- 📊 Watermark Quality Metrics ---")
    print(f"  - SSIM: {ssim_val.item():.4f}  (Higher is better, 1.0 is perfect)")
    print(f"  - PSNR: {psnr_val.item():.2f} dB (Higher is better)")
    print(f"  - MSE:  {mse_val.item():.6f} (Lower is better)")
    print("------------------------------------")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = ArgumentParser(description="Script to calculate watermark quality metrics.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Add new arguments for the required paths
    parser.add_argument("--start_checkpoint", type=str, default="output/leaves_wm/chkpnt10000.pth",
                        help="Path to the base .pth checkpoint file.")
    parser.add_argument("--ply_path", type=str, default="output/leaves_attack/merged_gaussians/point_cloud/iteration_2000/point_cloud.ply",
                        help="Path to the .ply file for Gaussian initialization (e.g., .../point_cloud.ply).")
    parser.add_argument("--gt_watermark_path", type=str, default="/sda1/qyzeng/GS-Hider/water/AAAI/AAAI_logo.jpg",
                        help="Path to the ground truth watermark PNG file.")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    # --- Execute the metrics calculation function ---
    calculate_watermark_metrics(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        checkpoint_path=args.start_checkpoint,
        ply_path=args.ply_path,
        gt_watermark_path=args.gt_watermark_path
    )

    print("\nScript finished. ✅")
