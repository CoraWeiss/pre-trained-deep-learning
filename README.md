# Image Classifier

A Python-based image classification tool that uses Google's EfficientNetV2 model to classify images according to ImageNet categories.

## Features

- Uses TensorFlow and TensorFlow Hub for efficient image classification
- Leverages pre-trained EfficientNetV2-B0 model
- Processes both single images and entire directories
- Outputs classification results to CSV with confidence scores
- Automatically downloads and caches ImageNet labels
- Progress tracking for batch processing

## Prerequisites

```bash
pip install tensorflow tensorflow-hub requests
```

## Usage

### Basic Usage

1. Place the script in a directory with your images
2. Run the script:

```bash
python image_classifier.py
```

The script will:
- Process all JPG/JPEG images in the current directory
- Generate a CSV file with results (format: `classifications_YYYYMMDD_HHMMSS.csv`)
- Show progress as it processes images

### Class Usage

You can also import and use the `ImageClassifier` class in your own code:

```python
from image_classifier import ImageClassifier

# Initialize the classifier
classifier = ImageClassifier()

# Classify a single image
result = classifier.classify_image('path/to/image.jpg')
print(result)

# Process an entire directory
classifier.process_directory('path/to/directory')
```

## Output Format

The script generates a CSV file with the following columns:
- `image`: Filename of the processed image
- `subject`: Predicted classification label
- `confidence`: Confidence score (0-1) for the prediction

## Example Output

### Command Line Execution
When you run the script, you'll see progress updates like this:
```bash
$ python image_classifier.py
Loading model...
Progress: 1/5 (20.0%)
Progress: 2/5 (40.0%)
Progress: 3/5 (60.0%)
Progress: 4/5 (80.0%)
Progress: 5/5 (100.0%)
Results saved to classifications_20240218_143022.csv
```

### Output CSV File
The generated CSV file (`classifications_20240218_143022.csv`) will look like this:
```csv
image,subject,confidence
cat.jpg,tabby cat,0.92
dog.jpg,golden retriever,0.87
bird.jpg,house finch,0.78
car.jpg,sports car,0.95
flower.jpg,daisy,0.89
```

### Python Code Output
When using the classifier in your own code:
```python
>>> from image_classifier import ImageClassifier
>>> classifier = ImageClassifier()
Loading model...
>>> result = classifier.classify_image('cat.jpg')
>>> print(result)
{
    'image': 'cat.jpg',
    'subject': 'tabby cat',
    'confidence': 0.92
}
```

## Technical Details

### Model Information
- Uses EfficientNetV2-B0 model from TensorFlow Hub
- Input images are resized to 224x224 pixels
- Pixel values are normalized to 0-1 range
- Predictions are made against 1000 ImageNet classes

### Image Processing
- Supports JPG/JPEG formats
- Images are automatically resized and normalized
- Error handling for corrupted or invalid images
- Memory-efficient processing of one image at a time

## Error Handling

The script includes error handling for:
- Missing image files
- Corrupted images
- Network issues during model/label download
- Invalid directories
- File permission issues

Failed image classifications are logged but don't halt batch processing.

## Limitations

- Only processes JPG/JPEG images
- Requires internet connection for first-time setup
- Classification limited to ImageNet categories
- Single-label classification only (no multi-label support)

## License

This project uses the EfficientNetV2 model which is licensed under the Apache License 2.0.

## Contributing

Feel free to submit issues and enhancement requests!
