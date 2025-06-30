import cv2
import numpy as np
import os
import shutil
import argparse

def align_images(thermal_path, rgb_path, min_matches=4):
    """
    Align thermal image to RGB image using feature matching while preserving aspect ratio and original colors.
    Returns aligned thermal image with the same dimensions as the RGB image.
    """
    # Read images - keep thermal in color
    img_rgb = cv2.imread(rgb_path)
    img_thermal_color = cv2.imread(thermal_path)  # Read in color
    
    if img_rgb is None or img_thermal_color is None:
        raise ValueError("Could not read one or both input images")
    
    # Convert RGB to grayscale for feature detection
    gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Create grayscale version of thermal for feature detection
    gray_thermal = cv2.cvtColor(img_thermal_color, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions
    h_rgb, w_rgb = gray_rgb.shape[:2]
    h_therm, w_therm = gray_thermal.shape[:2]
    
    # Calculate scaling factors
    scale_x = w_rgb / w_therm
    scale_y = h_rgb / h_therm
    
    # Use the smaller scale to ensure the thermal image fits within RGB dimensions
    scale = min(scale_x, scale_y)
    
    # Calculate new dimensions (maintain aspect ratio)
    new_w = int(w_therm * scale)
    new_h = int(h_therm * scale)
    new_size = (new_w, new_h)
    
    # Resize both color and grayscale thermal images
    img_thermal_resized = cv2.resize(gray_thermal, new_size, interpolation=cv2.INTER_LINEAR)
    img_thermal_color_resized = cv2.resize(img_thermal_color, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Calculate the scale to fit the thermal image within the RGB dimensions
    # while maintaining aspect ratio and adding smaller margins
    margin_percent = 0.03  # 3% margin on each side
    margin_w = int(w_rgb * margin_percent)
    margin_h = int(h_rgb * margin_percent)
    
    # Calculate scale with proportional margins
    scale = min((w_rgb - 2 * margin_w) / new_w, 
                (h_rgb - 2 * margin_h) / new_h)
    
    # Apply scaling to get new dimensions
    new_w = int(new_w * scale)
    new_h = int(new_h * scale)
    
    # Resize the thermal images with the new dimensions
    img_thermal_resized = cv2.resize(gray_thermal, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_thermal_color_resized = cv2.resize(img_thermal_color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding to center the image with proportional margins
    pad_top = (h_rgb - new_h) // 2
    pad_bottom = h_rgb - new_h - pad_top
    pad_left = (w_rgb - new_w) // 2
    pad_right = w_rgb - new_w - pad_left
    
    # Ensure minimum margins are maintained
    min_margin = 10  # 10px minimum margin
    pad_top = max(pad_top, min_margin)
    pad_bottom = max(pad_bottom, min_margin)
    pad_left = max(pad_left, min_margin)
    pad_right = max(pad_right, min_margin)
    
    # Print debug info
    print(f"  - Thermal size: {new_w}x{new_h}, RGB size: {w_rgb}x{h_rgb}")
    print(f"  - Padding - Top: {pad_top}, Bottom: {pad_bottom}, Left: {pad_left}, Right: {pad_right}")
    
    # Add padding to match RGB dimensions
    try:
        # For feature detection (grayscale)
        img_thermal_padded = cv2.copyMakeBorder(
            img_thermal_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        # For final output (color)
        img_thermal_color_padded = cv2.copyMakeBorder(
            img_thermal_color_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black border for color image
        )
    except Exception as e:
        print(f"  - Error during padding: {str(e)}")
        print(f"  - RGB dimensions: {h_rgb}x{w_rgb}, Thermal dimensions: {h_therm}x{w_therm}")
        print(f"  - New size: {new_h}x{new_w}, Paddings: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        raise
    
    # Initialize feature detector (ORB works well for thermal-RGB)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    
    # Find keypoints and descriptors on padded grayscale thermal
    kp1, des1 = orb.detectAndCompute(img_thermal_padded, None)
    kp2, des2 = orb.detectAndCompute(gray_rgb, None)
    
    if des1 is None or des2 is None or len(des1) < min_matches or len(des2) < min_matches:
        raise RuntimeError(f"Insufficient features detected: {len(des1) if des1 is not None else 0} vs {len(des2) if des2 is not None else 0}")
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"  - Found {len(matches)} matches")
    
    if len(matches) < min_matches:
        raise RuntimeError(f"Insufficient matches found: {len(matches)}/{min_matches}")
    
    # Get matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calculate center points for both images
    center_rgb = np.array([w_rgb/2, h_rgb/2])
    center_therm = np.array([new_w/2 + pad_left, new_h/2 + pad_top])
    
    # Calculate the offset needed to align the centers
    tx = center_rgb[0] - center_therm[0]
    ty = center_rgb[1] - center_therm[1]
    
    # Create translation matrix (only translation, no rotation or scaling)
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    
    print(f"  - Using center alignment only (no warping)")
    
    # Apply translation to the color padded thermal image
    aligned_thermal_color = cv2.warpAffine(
        img_thermal_color_padded,
        M,
        (w_rgb, h_rgb),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # Black border
    )
    
    # Create mask from the grayscale warped image
    warped_gray = cv2.warpAffine(
        img_thermal_padded,
        M,
        (w_rgb, h_rgb),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    _, mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
    
    return aligned_thermal_color, mask

def main(input_dir, output_dir):
    """
    Process all image pairs in input directory.
    Saves aligned thermal and RGB images to output directory.
    The thermal image will be warped to match the RGB image dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all thermal images
    thermal_images = [f for f in os.listdir(input_dir) if f.endswith('_T.JPG')]
    
    for thermal_file in thermal_images:
        base_name = thermal_file.split('_T.JPG')[0]
        rgb_file = f"{base_name}_Z.JPG"
        rgb_path = os.path.join(input_dir, rgb_file)
        
        if not os.path.exists(rgb_path):
            print(f"Missing RGB image for {thermal_file}. Skipping.")
            continue
        
        try:
            print(f"Processing {base_name}...")
            thermal_path = os.path.join(input_dir, thermal_file)
            
            # Verify input images exist and are readable
            if not os.path.exists(thermal_path):
                print(f"  - Error: Thermal image not found at {thermal_path}")
                continue
                
            if not os.path.exists(rgb_path):
                print(f"  - Error: RGB image not found at {rgb_path}")
                continue
            
            # Read the RGB image first to verify it's valid
            img_rgb = cv2.imread(rgb_path)
            if img_rgb is None:
                print(f"  - Error: Failed to read RGB image at {rgb_path}")
                continue
            
            try:
                # Align thermal to RGB (will have same dimensions as RGB image)
                aligned_thermal, _ = align_images(thermal_path, rgb_path)
                
                # Verify the aligned thermal image is valid
                if aligned_thermal is None or aligned_thermal.size == 0:
                    print(f"  - Error: Aligned thermal image is empty for {base_name}")
                    continue
                    
                # Additional validation: check if the aligned image is mostly black
                if np.mean(aligned_thermal) < 5:  # Arbitrary threshold
                    print(f"  - Warning: Aligned thermal image for {base_name} is mostly black")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output filenames
                thermal_output = os.path.join(output_dir, f"{base_name}_AT.JPG")
                rgb_output = os.path.join(output_dir, f"{base_name}_AC.JPG")
                
                # Save the aligned thermal image
                if not cv2.imwrite(thermal_output, aligned_thermal):
                    print(f"  - Error: Failed to save aligned thermal image to {thermal_output}")
                    continue
                
                # Save the original RGB image (not cropped)
                if not cv2.imwrite(rgb_output, img_rgb):
                    print(f"  - Error: Failed to save RGB image to {rgb_output}")
                    # Remove the partially saved thermal image if RGB save failed
                    if os.path.exists(thermal_output):
                        os.remove(thermal_output)
                    continue
                
                print(f"  - Successfully saved aligned images for {base_name}")
                
            except Exception as e:
                print(f"  - Error during alignment for {base_name}: {str(e)}")
                print("  - This usually happens when there aren't enough matching features between the images.")
                print("  - Try checking if the image pair is correctly matched or if the scene has enough distinctive features.")
                continue
                
                # Create output filenames
                thermal_output = os.path.join(output_dir, f"{base_name}_AT.JPG")
                rgb_output = os.path.join(output_dir, f"{base_name}_AC.JPG")
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Save results with error checking
                try:
                    # Ensure we have a valid image to save
                    if aligned_thermal is None or aligned_thermal.size == 0:
                        print(f"  - Error: Empty aligned thermal image for {thermal_file}")
                        continue
                        
                    # Convert to color if needed
                    if len(aligned_thermal.shape) == 2:  # Grayscale
                        aligned_thermal = cv2.cvtColor(aligned_thermal, cv2.COLOR_GRAY2BGR)
                    
                    # Ensure image is in correct format (8-bit, 3 channels)
                    if aligned_thermal.dtype != np.uint8:
                        aligned_thermal = cv2.normalize(aligned_thermal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Save the aligned thermal image
                    if not cv2.imwrite(thermal_output, aligned_thermal):
                        print(f"  - Error: Failed to save aligned thermal image to {thermal_output}")
                        continue
                        
                    # Save the RGB image
                    if not cv2.imwrite(rgb_output, img_rgb):
                        print(f"  - Error: Failed to save RGB image to {rgb_output}")
                        continue
                        
                    print(f"  - Successfully saved aligned images for {base_name}")
                    
                except Exception as e:
                    print(f"  - Error processing {thermal_file}: {str(e)}")
                    continue
                
                # Save the original RGB image (not cropped)
                if not cv2.imwrite(rgb_output, img_rgb):
                    print(f"  - Error: Failed to save RGB image to {rgb_output}")
                    # Remove the partially saved thermal image if RGB save failed
                    if os.path.exists(thermal_output):
                        os.remove(thermal_output)
                    continue
                
                print(f"  - Successfully processed and saved {base_name}")
                
            except Exception as e:
                print(f"  - Error during alignment for {base_name}: {str(e)}")
                continue
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, 
                        help="Input directory with image pairs")
    parser.add_argument("--output", type=str, default="task_1_output",
                        help="Output directory for results")
    args = parser.parse_args()
    
    main(args.input, args.output)