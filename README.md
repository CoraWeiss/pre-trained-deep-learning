# Image Analyzer

A comprehensive Python-based image analysis tool that combines object detection, classification, and feature extraction using TensorFlow Hub models.

## Features

- **Multiple Analysis Capabilities:**
  - Object detection using SSD MobileNet
  - Image classification using EfficientNetV2/MobileNet
  - Feature extraction for advanced analysis
- **Flexible Processing:**
  - Analyzes single images or entire directories
  - Detailed logging of the analysis process
  - Configurable confidence thresholds
- **Robust Implementation:**
  - Comprehensive error handling
  - Progress tracking with intermediate result saving
  - Detailed logging system
  - Memory-efficient processing
- **Rich Output:**
  - Generates detailed JSON results
  - Provides summary statistics of detected objects
  - Lists top classifications by confidence

## Prerequisites

```bash
pip install tensorflow tensorflow-hub pillow numpy requests
```

## Usage

### Basic Usage

1. Place the script in a directory with your images
2. Run the script:

```bash
python image_analyzer.py
```

The script will:
- Process all image files in the current directory
- Generate a JSON file with results (format: `image_analysis_YYYYMMDD_HHMMSS.json`)
- Create a log file with detailed processing information
- Show progress as it processes images

### Advanced Usage

You can also import and use the `ImageAnalyzer` class in your own code:

```python
from image_analyzer import ImageAnalyzer

# Initialize the analyzer with custom confidence threshold
analyzer = ImageAnalyzer(confidence_threshold=0.4)

# Analyze a single image
result = analyzer.analyze_image('path/to/image.jpg')
print(result)

# Process an entire directory with custom batch size
results = analyzer.analyze_directory('path/to/directory', batch_size=20)

# Get top classifications from results
top_classes = analyzer.get_top_classifications(results, top_n=5)
for item in top_classes:
    print(f"{item['image']}: {item['subject']} (confidence: {item['confidence']:.4f})")
```

## Output Format

The script generates a JSON file with comprehensive analysis for each image:

```json
{
  "example.jpg": {
    "image_info": {
      "filename": "example.jpg",
      "size": [800, 600],
      "mode": "RGB",
      "format": "JPEG"
    },
    "objects_detected": [
      {
        "object": "person",
        "confidence": 0.92,
        "location": [0.1, 0.2, 0.5, 0.8]
      },
      {
        "object": "car",
        "confidence": 0.87,
        "location": [0.6, 0.3, 0.9, 0.7]
      }
    ],
    "classification": {
      "main_subject": "street scene",
      "confidence": 0.79
    },
    "feature_vector_length": 1280
  }
}
```

## Technical Details

### Models Used

- **Object Detection:** SSD MobileNet v2 (fast, general-purpose detection)
- **Classification:** MobileNet v2 or EfficientNet v2 (configurable)
- **Feature Extraction:** MobileNet v2 feature vector

### Image Processing

- Supports JPG, JPEG, PNG, and GIF formats
- Images are automatically resized and normalized for each model
- Error handling for corrupted or invalid images
- Memory-efficient processing with batched saving

### Logging System

- Comprehensive logging to both console and file
- Detailed error messages and tracebacks
- Progress updates during batch processing
- Summary statistics upon completion

## Error Handling

The analyzer includes robust error handling for:

- Missing or corrupted image files
- Model loading failures
- Preprocessing issues
- File access or permission problems
- Network connectivity issues

Failed image analyses are logged but don't halt batch processing.

## Performance Considerations

- Processing large images or directories can be memory-intensive
- Feature vectors are summarized by default to save memory
- Intermediate results are saved periodically to prevent data loss
- Models are loaded once and reused for efficiency

## Customization Options

The analyzer provides several customization options:

- Adjustable confidence threshold for object detection
- Configurable batch size for directory processing
- Option to include or exclude full feature vectors
- Alternative models can be specified in the initialization

## Limitations

- Requires TensorFlow and sufficient RAM for model loading
- Classification limited to ImageNet categories
- Internet connection required for first-time model download
- Processing speed depends on hardware capabilities

## License

This project uses TensorFlow Hub models which are licensed under the Apache License 2.0.

## Contributing

Feel free to submit issues and enhancement requests!
