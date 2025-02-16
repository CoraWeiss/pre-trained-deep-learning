# AI Models in Image Analysis Pipeline

## Overview

The image analysis pipeline uses three pre-trained deep learning models from TensorFlow Hub, each serving a specific purpose in the analysis process. All models are automatically downloaded and cached when first run.

## 1. Object Detection: SSD MobileNet V2

**Model URL:** `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2`

**Purpose:** Detects and localizes multiple objects within a single image.

**Technical Details:**
- Architecture: Single Shot Detector (SSD) with MobileNet V2 backbone
- Input: RGB images (any size, automatically resized)
- Output: Multiple objects with bounding boxes and confidence scores
- Confidence Threshold: 0.5 (configurable)

**Implementation:**
```python
def detect_objects(self, image_path):
    # Processes image through SSD MobileNet
    # Returns list of detected objects with:
    # - Object class name
    # - Confidence score
    # - Bounding box coordinates [y1, x1, y2, x2]
```

**Use Cases:**
- Multiple object detection in complex scenes
- Spatial relationship analysis
- Object counting and localization

## 2. Image Classification: EfficientNet V2 B0

**Model URL:** `https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2`

**Purpose:** Identifies the primary subject or main content of the image.

**Technical Details:**
- Architecture: EfficientNet V2 B0
- Training Dataset: ImageNet-1K
- Classes: 1000 categories
- Input Size: 224x224 pixels
- Output: Class probabilities across all categories

**Implementation:**
```python
def classify_image(self, image_path):
    # Returns dictionary with:
    # - Main subject (highest probability class)
    # - Confidence score
```

**Key Features:**
- State-of-the-art accuracy/efficiency tradeoff
- Robust to various image conditions
- Handles diverse subject matter

## 3. Feature Extraction: EfficientNet V2 B0 Feature Vector

**Model URL:** `https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2`

**Purpose:** Extracts high-level visual features for scene understanding and similarity comparison.

**Technical Details:**
- Architecture: Modified EfficientNet V2 B0 (pre-classification layer)
- Output: 1280-dimensional feature vector
- Input Size: 224x224 pixels
- Vector Type: Dense numerical representation

**Implementation:**
```python
def extract_features(self, image_path):
    # Returns:
    # - 1280-dimensional numpy array
    # - Represents high-level image features
```

**Applications:**
- Image similarity comparison
- Scene understanding
- Content-based image retrieval
- Transfer learning base

## Model Pipeline Flow

1. **Image Loading**
   - Load image using TensorFlow I/O
   - Convert to appropriate format for each model

2. **Parallel Processing**
   ```
   Input Image
   ├── Object Detection (SSD MobileNet)
   ├── Classification (EfficientNet)
   └── Feature Extraction (EfficientNet)
   ```

3. **Result Aggregation**
   - Combine outputs from all models
   - Format into structured JSON
   - Save with image metadata

## Performance Considerations

- **Memory Usage:** Models are loaded once and kept in memory
- **Processing Speed:** ~2-5 seconds per image on CPU
- **GPU Acceleration:** Automatic if available
- **Batch Processing:** Sequential to manage memory usage

## Model Limitations

1. **SSD MobileNet V2**
   - Fixed confidence threshold
   - Limited to common object categories
   - Resolution-dependent accuracy

2. **EfficientNet V2 B0**
   - Limited to 1000 ImageNet classes
   - Single-label classification only
   - Fixed input resolution

3. **Feature Extractor**
   - Fixed feature vector dimensionality
   - ImageNet-biased features
   - No direct semantic meaning
