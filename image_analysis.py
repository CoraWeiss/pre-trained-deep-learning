import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("image_analyzer.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self, confidence_threshold=0.3):
        """
        Initialize the ImageAnalyzer with multiple models for comprehensive image analysis.
        
        Args:
            confidence_threshold: Minimum confidence score to report for detections (default: 0.3)
        """
        self.confidence_threshold = confidence_threshold
        logger.info("Loading models...")
        
        try:
            # Load object detection model (SSD MobileNet)
            # Note: SSD MobileNet is optimized for speed but may have lower accuracy
            # Consider using EfficientDet or Faster R-CNN for better results
            self.detector = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
            
            # Load image classification model (MobileNet by default)
            # You can experiment with different models:
            # - EfficientNet: https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2
            # - ResNet: https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5
            self.classifier = hub.load('https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4')
            
            # Load feature extraction model for scene understanding
            self.feature_extractor = hub.load('https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4')
            
            # Load ImageNet labels
            self.imagenet_labels = self.get_imagenet_labels()
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def get_imagenet_labels(self):
        """Load or download ImageNet labels"""
        labels_path = 'imagenet_labels.txt'
        
        # If labels file doesn't exist, download it
        if not os.path.exists(labels_path):
            logger.info("Downloading ImageNet labels...")
            url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise exception for HTTP errors
                labels = [line.strip() for line in response.text.splitlines()]
                
                # Save labels locally for future use
                with open(labels_path, 'w') as f:
                    f.write('\n'.join(labels))
                
                return labels
            except Exception as e:
                logger.error(f"Error downloading labels: {str(e)}")
                # Fallback to a minimal set of labels if download fails
                return ["background"] + [f"class_{i}" for i in range(1, 1001)]
        
        # If file exists, read it
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f]
        except Exception as e:
            logger.error(f"Error reading labels file: {str(e)}")
            return ["background"] + [f"class_{i}" for i in range(1, 1001)]

    def load_and_preprocess_image(self, image_path, target_size=(224, 224), normalize=True):
        """
        Load and preprocess image for model input
        
        Args:
            image_path: Path to the image file
            target_size: Tuple (height, width) for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            
            # Add batch dimension if needed
            if img.shape.rank == 3:
                img = tf.expand_dims(img, axis=0)
            
            # Resize image
            img = tf.image.resize(img, target_size)
            
            # Convert to float and normalize
            if normalize:
                img = tf.cast(img, tf.float32) / 255.0
            
            return img
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            # Return a placeholder black image in case of error
            return tf.zeros([1, target_size[0], target_size[1], 3], dtype=tf.float32)

    def detect_objects(self, image_path):
        """
        Detect objects in the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected objects with confidence scores and bounding boxes
        """
        try:
            # Load image without resizing for detection (let the model handle it)
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            
            # Convert to float [0,1]
            img = tf.cast(img, tf.float32) / 255.0
            
            # Add batch dimension if needed
            if img.shape.rank == 3:
                img = tf.expand_dims(img, axis=0)
            
            # Run detection
            result = self.detector(img)
            
            objects_found = []
            
            # Process results - handle detections above threshold
            detection_scores = result['detection_scores'][0].numpy()
            detection_classes = result['detection_class_names'][0].numpy()
            detection_boxes = result['detection_boxes'][0].numpy()
            
            for i, score in enumerate(detection_scores):
                if score > self.confidence_threshold:
                    class_name = detection_classes[i].decode('utf-8')
                    objects_found.append({
                        'object': class_name,
                        'confidence': float(score),
                        'location': detection_boxes[i].tolist()
                    })
            
            logger.info(f"Detected {len(objects_found)} objects in {image_path}")
            return objects_found
        except Exception as e:
            logger.error(f"Error in detect_objects for {image_path}: {str(e)}")
            return []

    def classify_image(self, image_path):
        """
        Classify the main subject of the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with main subject and confidence score
        """
        try:
            # Preprocess image for classification
            img = self.load_and_preprocess_image(image_path)
            
            # Run classification
            predictions = self.classifier(img)
            predictions = tf.squeeze(predictions)
            
            # Get top prediction
            predicted_class = tf.argmax(predictions)
            confidence = float(tf.nn.softmax(predictions)[predicted_class])
            
            # Get class name (offset by 1 as ImageNet labels typically start at index 1)
            class_idx = int(predicted_class)
            if len(self.imagenet_labels) > 1000:  # If using the full ImageNet labels
                class_name = self.imagenet_labels[class_idx]
            else:
                # Adjust for potential label list format variations
                class_name = self.imagenet_labels[max(0, class_idx - 1)]
            
            logger.info(f"Classified {image_path} as {class_name} with confidence {confidence:.4f}")
            return {
                'main_subject': class_name,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error in classify_image for {image_path}: {str(e)}")
            return {'main_subject': 'unknown', 'confidence': 0.0}

    def extract_features(self, image_path):
        """
        Extract high-level features from the image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Preprocess image for feature extraction
            img = self.load_and_preprocess_image(image_path)
            
            # Extract features
            features = self.feature_extractor(img)
            features = tf.squeeze(features)
            
            logger.info(f"Extracted {features.shape[0]} features from {image_path}")
            return features.numpy()
        except Exception as e:
            logger.error(f"Error in extract_features for {image_path}: {str(e)}")
            return np.zeros(1280)  # Return zero vector in case of error

    def analyze_image(self, image_path):
        """
        Perform comprehensive analysis of an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing image: {image_path}")
        
        try:
            # Get basic image info
            with Image.open(image_path) as img:
                image_info = {
                    'filename': os.path.basename(image_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format
                }

            # Detect objects
            objects = self.detect_objects(image_path)
            
            # Classify main subject
            classification = self.classify_image(image_path)
            
            # Extract feature vector (optional - can be memory intensive)
            features = self.extract_features(image_path)
            
            analysis = {
                'image_info': image_info,
                'objects_detected': objects,
                'classification': classification,
                # Store only feature vector length to save space
                'feature_vector_length': len(features),
                # Uncomment to include full feature vector (large)
                # 'feature_vector': features.tolist()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {str(e)}")
            return {
                'error': str(e),
                'image_path': image_path
            }

    def analyze_directory(self, directory='.', save_results=True, batch_size=10):
        """
        Analyze all images in a directory
        
        Args:
            directory: Directory containing images
            save_results: Whether to save results to a JSON file
            batch_size: Number of images to process before saving intermediate results
            
        Returns:
            Dictionary with analysis results for all images
        """
        results = {}
        
        # Get all image files
        image_files = [f for f in os.listdir(directory) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        total_images = len(image_files)
        logger.info(f"Found {total_images} images to analyze in {directory}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'image_analysis_{timestamp}.json'
        
        # Process images with progress reporting
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(directory, img_file)
            try:
                logger.info(f"Processing image {idx}/{total_images}: {img_file}")
                results[img_file] = self.analyze_image(img_path)
                
                # Print progress percentage
                progress = (idx / total_images) * 100
                logger.info(f"Overall progress: {progress:.1f}%")
                
                # Save intermediate results periodically
                if save_results and idx % batch_size == 0:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved intermediate results to {output_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
                results[img_file] = {'error': str(e)}
        
        # Save final results
        if save_results:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Analysis results saved to {output_file}")
        
        return results

    def get_top_classifications(self, results, top_n=10):
        """
        Get the most confident classifications from results
        
        Args:
            results: Dictionary of analysis results
            top_n: Number of top results to return
            
        Returns:
            List of top classifications with confidence scores
        """
        classifications = []
        
        for img_name, analysis in results.items():
            if analysis and 'classification' in analysis:
                classifications.append({
                    'image': img_name,
                    'subject': analysis['classification']['main_subject'],
                    'confidence': analysis['classification']['confidence']
                })
        
        # Sort by confidence (descending)
        classifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        return classifications[:top_n]

def main():
    """Main function to run the image analyzer"""
    logger.info("Initializing Image Analyzer...")
    analyzer = ImageAnalyzer(confidence_threshold=0.4)  # Increased confidence threshold
    
    # Use a specific directory or current directory
    image_dir = '.'  # Replace with your images directory
    logger.info(f"Starting analysis of directory: {image_dir}")
    
    results = analyzer.analyze_directory(image_dir)
    
    # Print summary of findings
    logger.info("\nAnalysis Summary:")
    
    # Get and print top classifications
    top_classes = analyzer.get_top_classifications(results, top_n=5)
    logger.info("\nTop 5 Most Confident Classifications:")
    for i, item in enumerate(top_classes, 1):
        logger.info(f"{i}. {item['image']}: {item['subject']} (confidence: {item['confidence']:.4f})")
    
    # Count object types
    object_counts = {}
    for img_name, analysis in results.items():
        if analysis and 'objects_detected' in analysis:
            for obj in analysis['objects_detected']:
                obj_name = obj['object']
                if obj_name in object_counts:
                    object_counts[obj_name] += 1
                else:
                    object_counts[obj_name] = 1
    
    # Print top detected objects
    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 Detected Objects:")
    for obj_name, count in top_objects:
        logger.info(f"- {obj_name}: {count} instances")

if __name__ == "__main__":
    main()
