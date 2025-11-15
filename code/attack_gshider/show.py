# import os
# import sys
# import torch
# import torch.nn as nn
# import numpy as np
# from argparse import ArgumentParser
# from plyfile import PlyData
#
# # Assuming all custom modules are accessible from the script's location
# from scene import Scene, GaussianModel
# from gaussian_renderer import render
# from arguments import ModelParams, OptimizationParams, PipelineParams
# from decoder import SimpleCNN, WatermarkCNN
# from utils.general_utils import safe_state
#
# # Set the visible CUDA device
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
#
# # --- The function you provided to load PLY files ---
# # This function is correct and remains unchanged.
# def load_ply_as_initialization(ply_path, dataset, opt):
#     """
#     Loads Gaussian model parameters from a PLY file.
#     This version is specifically designed for models with 'semantic_' features
#     instead of traditional color features.
#     """
#     gaussians = GaussianModel(dataset.sh_degree)
#
#     try:
#         plydata = PlyData.read(ply_path)
#     except Exception as e:
#         print(f"[ERROR] Could not read PLY file at: {ply_path}. Error: {e}")
#         sys.exit(1)
#
#     data = plydata.elements[0].data
#
#     # Check for required attributes
#     required_attrs = ['x', 'y', 'z', 'opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
#     for attr in required_attrs:
#         if attr not in data.dtype.names:
#             raise ValueError(f"ERROR: PLY file '{ply_path}' is missing the required attribute '{attr}'.")
#
#     xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
#     opacities = data['opacity'].reshape(-1, 1)
#     scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
#     rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()
#
#     # Dynamically find and load semantic features
#     semantic_feature_names = [name for name in data.dtype.names if name.startswith('semantic_')]
#     if not semantic_feature_names:
#         raise ValueError(f"ERROR: PLY file '{ply_path}' does not contain any 'semantic_' feature fields.")
#
#     semantic_dim = len(semantic_feature_names)
#     print(f"[INFO] Found {semantic_dim}-dimensional semantic features in the PLY file.")
#
#     semantic_features = np.stack([data[name] for name in semantic_feature_names]).transpose()
#     semantic_features = semantic_features.reshape(-1, 1, semantic_dim)
#
#     # Assign to GaussianModel as torch tensors
#     gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
#
#     # Note: The attribute name must match what is used in your custom GaussianModel
#     gaussians._semantic_feature = nn.Parameter(
#         torch.tensor(semantic_features, dtype=torch.float, device="cuda").requires_grad_(True))
#
#     # Initialize other required tensors for rendering
#     gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
#     gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
#     gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
#
#     # Call training_setup to initialize optimizers, etc.
#     gaussians.training_setup(opt)
#
#     return gaussians
#
#
# def generate_images_from_ply(dataset, opt, pipe, ply_path, decoder_checkpoint_path):
#     """
#     Loads a 3DGS model from a .ply file and decoders from checkpoints,
#     then performs a rendering pass to generate 'rgb_image' and 'water_image'.
#     """
#     # --- 1. Load Gaussian Model from PLY and Initialize Scene ---
#     print("\n[INFO] Initializing scene and loading Gaussians from PLY file...")
#     if not os.path.exists(ply_path):
#         print(f"[ERROR] PLY file not found at: {ply_path}")
#         return
#
#     gaussians = GaussianModel(dataset.sh_degree)
#     (model_params, _) = torch.load(decoder_checkpoint_path, map_location="cuda", weights_only=False)
#     gaussians.restore(model_params, opt)
#
#     # Use the provided function to load gaussians from the PLY file
#     # gaussians = load_ply_as_initialization(ply_path, dataset, opt)
#     scene = Scene(dataset, gaussians, shuffle=False)
#
#     # --- 2. Load Decoder Networks ---
#     print(f"\n[INFO] Loading decoder networks using base path: {decoder_checkpoint_path}")
#
#     # Initialize the two decoder networks
#     imagenet = SimpleCNN().cuda()
#     waternet = WatermarkCNN().cuda()
#
#     # Construct paths for the decoder networks based on the provided checkpoint file
#     net_checkpoint_path = decoder_checkpoint_path.replace('.pth', '_net.pth')
#     waternet_checkpoint_path = decoder_checkpoint_path.replace('.pth', '_waternet.pth')
#
#     # Load the RGB decoder (imagenet)
#     if os.path.exists(net_checkpoint_path):
#         imagenet.load_state_dict(torch.load(net_checkpoint_path, map_location="cuda", weights_only=False))
#         print(f"[INFO] Successfully loaded RGB decoder from: {net_checkpoint_path}")
#     else:
#         print(f"[ERROR] RGB decoder checkpoint not found at: {net_checkpoint_path}")
#         return
#
#     # Load the watermark decoder (waternet)
#     if os.path.exists(waternet_checkpoint_path):
#         waternet.load_state_dict(torch.load(waternet_checkpoint_path, map_location="cuda", weights_only=False))
#         print(f"[INFO] Successfully loaded Watermark decoder from: {waternet_checkpoint_path}")
#     else:
#         print(f"[ERROR] Watermark decoder checkpoint not found at: {waternet_checkpoint_path}")
#         return
#
#     # **CRITICAL STEP**: Set models to evaluation mode
#     imagenet.eval()
#     waternet.eval()
#
#     print("[INFO] All models set to evaluation mode.")
#
#     # --- 3. Render and Decode a Single View ---
#     with torch.no_grad():
#         viewpoint_cam = scene.getTrainCameras()[0]
#         print(f"\n[INFO] Selected camera '{viewpoint_cam.image_name}' for rendering.")
#
#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
#         # Step A: Render the 3D Gaussians to get the intermediate feature map
#         render_pkg = render(viewpoint_cam, gaussians, pipe, background)
#         feature_image = render_pkg["render"]
#
#         # Step B: Pass the feature map through both decoders
#         rgb_image = imagenet(feature_image.unsqueeze(0)).squeeze(0)
#         water_image = waternet(feature_image.unsqueeze(0)).squeeze(0)
#
#     # --- 4. Final Output ---
#     print("\n[SUCCESS] The target variables have been generated.")
#     print(f" - variable 'rgb_image':   Type={type(rgb_image)}, Shape={rgb_image.shape}, Device={rgb_image.device}")
#     print(
#         f" - variable 'water_image': Type={type(water_image)}, Shape={water_image.shape}, Device={water_image.device}")
#
#     # Example of how to save the output images
#     # from torchvision.utils import save_image
#     # save_image(rgb_image, "output_rgb_from_ply.png")
#     # save_image(water_image, "output_watermark_from_ply.png")
#
#
# if __name__ == "__main__":
#     # Set up a simplified argument parser
#     parser = ArgumentParser(description="Script to load a PLY model and trained decoders to generate images.")
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#
#     # MODIFIED: Argument to point to the PLY file
#     parser.add_argument("--ply_path", type=str, default="output/fern_wm/point_cloud/iteration_10000/point_cloud.ply",
#                         help="Path to the input PLY file with semantic features.")
#
#     # MODIFIED: Argument to point to the corresponding decoder checkpoint
#     parser.add_argument("--decoder_checkpoint", type=str, default="output/fern_wm/chkpnt10000.pth",
#                         help="Path to the base checkpoint file for loading decoders (e.g., output/fern_wm/chkpnt10000.pth).")
#
#     parser.add_argument("--quiet", action="store_true")
#
#     args = parser.parse_args(sys.argv[1:])
#     safe_state(args.quiet)
#
#     # Execute the generation function with the new arguments
#     generate_images_from_ply(
#         dataset=lp.extract(args),
#         opt=op.extract(args),
#         pipe=pp.extract(args),
#         ply_path=args.ply_path,
#         decoder_checkpoint_path=args.decoder_checkpoint
#     )
#
#     print("\nScript finished.")


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
from decoder import SimpleCNN, WatermarkCNN
from utils.general_utils import safe_state
from plyfile import PlyData
import numpy as np
import random

random_seed = 0  # 固定随机种子保证可复现性
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
# Set the visible CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def generate_images_from_checkpoint(dataset, opt, pipe, checkpoint_path):
    """
    Loads a 3DGS model and two decoders from a checkpoint, then performs
    a single rendering pass to generate the 'rgb_image' and 'water_image'.
    """
    # --- 1. Initialize Models and Scene ---
    print("\n[INFO] Initializing models and scene...")

    gaussians = GaussianModel(dataset.sh_degree) # hahaha
    scene = Scene(dataset, gaussians, shuffle=False) # Use shuffle=False for predictable camera order

    # Initialize the two decoder networks
    imagenet = SimpleCNN().cuda()
    waternet = WatermarkCNN().cuda()

    # --- 2. Load Pre-trained Weights ---
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found at: {checkpoint_path}")
        return

    print(f"\n[INFO] Loading all models from checkpoint: {checkpoint_path}")

    # Load the main Gaussian Model parameters
    # (model_params, _) = torch.load(checkpoint_path, map_location="cuda", weights_only=False)  # hahaha
    # gaussians.restore(model_params, opt)  #hahaha
    gaussians = load_ply_as_initialization(ply_path, dataset, opt)

    # Construct paths for the decoder networks based on the main checkpoint file
    net_checkpoint_path = checkpoint_path.replace('.pth', '_net.pth')
    waternet_checkpoint_path = checkpoint_path.replace('.pth', '_waternet.pth')

    # Load the RGB decoder (imagenet)
    if os.path.exists(net_checkpoint_path):
        imagenet.load_state_dict(torch.load(net_checkpoint_path, map_location="cuda", weights_only=False))
        print(f"[INFO] Successfully loaded RGB decoder from: {net_checkpoint_path}")
    else:
        print(f"[ERROR] RGB decoder checkpoint not found at: {net_checkpoint_path}")
        return

    # Load the watermark decoder (waternet)
    if os.path.exists(waternet_checkpoint_path):
        waternet.load_state_dict(torch.load(waternet_checkpoint_path, map_location="cuda", weights_only=False))
        print(f"[INFO] Successfully loaded Watermark decoder from: {waternet_checkpoint_path}")
    else:
        print(f"[ERROR] Watermark decoder checkpoint not found at: {waternet_checkpoint_path}")
        return

    # **CRITICAL STEP**: Set models to evaluation mode. This disables training-specific
    # layers like Dropout and ensures consistent output.
    imagenet.eval()
    waternet.eval()
    print("[INFO] All models set to evaluation mode.")

    # --- 3. Render and Decode a Single View ---
    # Everything is done within torch.no_grad() to save memory and computation,
    # as we are not training and don't need gradients.
    with torch.no_grad():
        # Select a camera viewpoint (e.g., the first training camera)
        viewpoint_cam = scene.getTrainCameras()[26]
        print(f"\n[INFO] Selected camera '{viewpoint_cam.image_name}' for rendering.")

        # Set up the background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Step A: Render the 3D Gaussians to get the intermediate feature map
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        feature_image = render_pkg["render"]

        # Step B: Pass the feature map through both decoders to get the final images
        # The decoders expect a batch dimension, so we add it with .unsqueeze(0)
        # and remove it afterward with .squeeze(0).
        rgb_image = imagenet(feature_image.unsqueeze(0)).squeeze(0)
        water_image = waternet(feature_image.unsqueeze(0)).squeeze(0)

    # --- 4. Final Output ---
    # The desired variables are now available.
    print("\n[SUCCESS] The target variables have been generated.")
    print(f" - variable 'rgb_image':   Type={type(rgb_image)}, Shape={rgb_image.shape}, Device={rgb_image.device}")
    print(f" - variable 'water_image': Type={type(water_image)}, Shape={water_image.shape}, Device={water_image.device}")

    # You can now use these variables for any further processing.
    # Example:
    from torchvision.utils import save_image
    save_image(rgb_image, "output_rgb.png")
    save_image(water_image, "output_watermark.png")


if __name__ == "__main__":
    # Set up a simplified argument parser
    parser = ArgumentParser(description="Script to load trained models and generate images.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default="output/room_wm/chkpnt10000.pth", help="Path to the checkpoint file (e.g., chkpnt30000.pth).")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    ply_path = 'output/room_reconstruct/point_cloud/iteration_10000/point_cloud.ply'  # 'output/fortress_wm/point_cloud/iteration_40000/point_cloud.ply'  'output/fortress_attack/merged_gaussians/point_cloud/iteration_2000/point_cloud.ply'

    # Execute the generation function
    generate_images_from_checkpoint(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        checkpoint_path=args.start_checkpoint
    )

    print("\nScript finished.")