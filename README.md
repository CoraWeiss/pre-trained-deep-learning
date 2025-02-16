# Image Analysis Pipeline

A comprehensive image analysis tool that uses multiple deep learning models to automatically analyze, classify, and extract information from images. The system processes images in batch and provides outputs in both CSV and JSON formats for easy analysis.

## Introduction to AI Models Used

This pipeline combines three specialized deep learning models, each designed to understand different aspects of image content:

### 1. Object Detection (SSD MobileNet V2)
Think of this model as our "object spotter." It scans the entire image and can find multiple objects simultaneously, telling us:
- What objects are present (e.g., "person," "car," "dog")
- How confident it is about each detection (e.g., 95% sure)
- Where exactly in the image each object is located (using coordinate boxes)

### 2. Image Classification (EfficientNet V2 B0)
This model acts as our "scene interpreter." It looks at the entire image and determines:
- The main subject or primary content
- How confident it is about this classification
- Categorizes from among 1000 different possible subjects

### 3. Feature Extraction (EfficientNet V2 B0)
This is our "technical analyzer." It creates a detailed numerical representation of the image by:
- Converting visual information into 1280 numbers
- Capturing abstract features like shapes, patterns, and textures
- Enabling technical comparisons between images

## Quick Start

1. Install dependencies:
```bash
pip install tensorflow tensorflow-hub pillow numpy requests
```

2. Place your images in the script directory

3. Run the analysis:
```bash
python image_analysis.py
```

## Output Files

The tool generates two types of output files:

### 1. CSV Output (`image_analysis_YYYYMMDD_HHMMSS.csv`)
Easy-to-read format for quick analysis:
```csv
image_name,main_subject,confidence,detected_objects
photo1.jpg,dog,0.95,"person(0.98); leash(0.85); bowl(0.76)"
photo2.jpg,cat,0.88,"person(0.92); couch(0.78); plant(0.65)"
```

### 2. JSON Output (`image_analysis_YYYYMMDD_HHMMSS.json`)
Detailed technical data including:
```json
{
  "image_name.jpg": {
    "image_info": {
      "size": [1024, 768],
      "format": "JPEG"
    },
    "objects_detected": [
      {
        "object": "person",
        "confidence": 0.98,
        "location": [0.1, 0.2, 0.8, 0.9]
      }
    ],
    "classification": {
      "main_subject": "dog",
      "confidence": 0.95
    },
    "feature_vector": [...]
  }
}
```

## Processing Performance

- **First Run**: 
  - Downloads models (~5-10 minutes)
  - Creates cache
  - Downloads label files
- **Subsequent Runs**: 
  - 2-5 seconds per image
  - GPU acceleration if available
  - Sequential processing for memory efficiency

## System Requirements

- Python 3.8+
- 8GB RAM recommended
- 1GB free disk space
- Internet connection for first run
- CUDA-compatible GPU (optional)

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Technical Details

### Model URLs
- Object Detection: `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2`
- Classification: `https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2`
- Feature Extraction: `https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2`

### Processing Pipeline
```
Input Image
│
├─► Object Detection
│   └─► Objects + Locations
│
├─► Classification
│   └─► Main Subject
│
└─► Feature Extraction
    └─► Feature Vector
```

## Error Handling

- Skips corrupted images
- Continues processing after errors
- Logs all errors
- Saves partial results

## Known Limitations

- Fixed confidence threshold (0.5) for object detection
- Limited to 1000 ImageNet categories
- Requires significant RAM for model loading
- Internet required for first run

## Folder Structure
```
.
├── image_analysis.py      # Main script
├── imagenet_labels.txt    # Downloaded on first run
├── images/               # Your image directory
├── image_analysis_*.csv   # Generated CSV output
└── image_analysis_*.json  # Generated JSON output
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- TensorFlow team for the pre-trained models
- TensorFlow Hub for model distribution
- ImageNet for training data and categories
