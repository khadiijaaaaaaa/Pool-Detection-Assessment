import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_pool(input_path, output_dir="output_results"):
    """
    Detects swimming pools, draws a blue contour, and saves coordinates.
    """
    # input path
    # If the file provided isn't found, try looking in 'input_images/'
    if not os.path.exists(input_path):
        potential_path = os.path.join("input_images", input_path)
        if os.path.exists(potential_path):
            input_path = potential_path
        else:
            print(f"Error: Could not find image at {input_path} or {potential_path}")
            return

    # output path
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We'll load our custom trained model: best.pt
    # Model: YOLOv8n-seg (Nano Segmentation)
    # Training: Custom dataset (226 images) trained for 50 epochs on Google Colab
    # Source: 'best.pt' file generated from the training run
    model_path = "best.pt" 
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Please place it in this directory.")
        return

    model = YOLO(model_path)

    # we'll Run Inference
    results = model.predict(source=input_path, conf=0.25, save=False, verbose=False) #conf=0.25 ignores low-confidence guesses
    result = results[0]

    # Load original image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return

    # Prepare outputs
    base_name = os.path.basename(input_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    
    # Process Detections
    coord_lines = []
    
    # Check if any masks were found
    if result.masks is not None:
        # Get polygon segments (the exact shape outline)
        # xyn returns normalized coordinates (0-1), xy returns pixels. 
        # Using pixels is usually safer for visual verification.
        masks_pixel = result.masks.xy 

        for i, mask in enumerate(masks_pixel):
            # Create a label
            line_content = f"Pool_{i}:"
            for point in mask:
                # Format: x,y x,y ...
                line_content += f" {int(point[0])},{int(point[1])}"
            coord_lines.append(line_content)

            # draw blue outline
            pts = np.array(mask, np.int32) # Convert mask points to int32 for OpenCV
            pts = pts.reshape((-1, 1, 2))
            
            # cv2.polylines draws the contour
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    else:
        print("No pools detected in this image.")

    # Save Output Image
    output_img_path = os.path.join(output_dir, f"output_{base_name}")
    cv2.imwrite(output_img_path, img)

    # Save Coordinates Text File
    output_txt_path = os.path.join(output_dir, f"coordinates_{file_name_no_ext}.txt")
    with open(output_txt_path, "w") as f:
        f.write("\n".join(coord_lines))

    print(f"Success! Processed {base_name}")
    print(f" - Image saved to: {output_img_path}")
    print(f" - Coordinates saved to: {output_txt_path}")

if __name__ == "__main__":
    # Setup Argument Parser for CLI
    parser = argparse.ArgumentParser(description="Detect swimming pools and output contours.")
    parser.add_argument("input_image", help="Path to the input aerial image")
    
    args = parser.parse_args()
    
    detect_pool(args.input_image)