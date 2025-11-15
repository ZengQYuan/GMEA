import cv2
from skimage.metrics import structural_similarity as ssim
import os

def calculate_ssim_between_images(image_path1, image_path2):
    """
    Reads two images from file paths and calculates their SSIM.
    """
    # Check if files exist
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print(f"Error: One or both image files not found.")
        return None

    # --- 1. Read Images ---
    # Use cv2.imread to load images as NumPy arrays
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # --- 2. Convert to Grayscale ---
    # SSIM is typically calculated on single-channel (grayscale) images.
    # The cv2.COLOR_BGR2GRAY is used because OpenCV loads images in BGR order.
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- 3. Check for Same Dimensions ---
    if img1_gray.shape != img2_gray.shape:
        print("Error: Images have different dimensions and cannot be compared.")
        # Optional: resize one image to match the other
        # h, w = img1_gray.shape
        # img2_gray = cv2.resize(img2_gray, (w, h))
        return None

    # --- 4. Calculate SSIM ---
    # The ssim function returns the SSIM value and optionally a difference image.
    # The data_range is the dynamic range of the input images (255 for 8-bit images).
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True, data_range=255)

    return ssim_value

if __name__ == '__main__':
    # Define the paths to your two images
    path1 = 'output_watermark1.png'
    path2 = 'output_watermark2.png'

    # Calculate and print the SSIM score
    similarity_score = calculate_ssim_between_images(path1, path2)

    if similarity_score is not None:
        # The SSIM value is a float between -1 and 1, where 1 means identical.
        print(f"🖼️ The SSIM between '{path1}' and '{path2}' is: {similarity_score:.4f}")
