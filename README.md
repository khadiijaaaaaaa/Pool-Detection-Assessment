# Swimming Pool Detection

This repository contains a CLI solution to detect swimming pools in aerial imagery. It generates a visual outline in blue and a text file containing the exact shape coordinates.

## Project Structure

```text
Sanadtech-Pool-Detection/
│
├── input_images/          # Contains the original sample aerial images
│   ├── 000000079.jpg
│   └── ...
│
├── output_results/        # Generated results (images + coordinates)
│   ├── output_000000079.jpg
│   ├── coordinates_000000079.txt
│   ├── output_000000216.jpg
│   ├── coordinates_000000216.txt
│   └── ...
│
├── pool_detector.py       # Main CLI script for detection
├── best.pt                # Trained YOLOv8 Segmentation model
└── requirements.txt       # Project dependencies
```

## Approach
To ensure accurate detection of **irregular shapes** (kidney, oval, L-shape) as requested, I used **Instance Segmentation** rather than standard object detection (bounding boxes).

- **Model:** YOLOv8 Nano Segmentation (`yolov8n-seg`)
- **Dataset:** Custom dataset of 226 aerial images (198 Train, 19 Val, 9 Test)
- **Training:** 50 Epochs

### Model Performance
The model achieved high accuracy on the validation set, effectively handling shadows and curved edges.
- **Precision:** 0.981
- **Recall:** 0.885
- **mAP50:** 0.922

## Setup & Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

2. **Run the Script**
    ```Bash
    python pool_detector.py input_images/000000216.jpg

3. **Check Output** The script will generate two files in `output_results/` directory:

    `output_000000216.jpg`: Image with the detected pool outlined in blue.

    `coordinates_000000216.txt`: Text file with polygon coordinates.

## Sample Results
1. **Irregular Shapes (Kidney Pool)** Successfully detects curved edges where bounding boxes would fail.
2. **Occlusions (Shadows) Successfully** detects the pool despite significant tree shadows.

Full results for all sample images can be found in the `output_results/` folder.
