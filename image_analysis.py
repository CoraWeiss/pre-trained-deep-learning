import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import requests

class ImageAnalyzer:
    def __init__(self):
        print("Loading models...")
        # Load object detection model (SSD MobileNet)
        self.detector = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
        
        # Load image classification model (EfficientNet)
        self.classifier = hub.load('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2')
        
        # Load feature extraction model for scene understanding
        self.feature_extractor = hub.load('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2')
        
        # Load or download ImageNet labels
        self.imagenet_labels = self.get_imagenet_labels()

    def get_imagenet_labels(self):
        """Load or download ImageNet labels"""
        labels_path = 'imagenet_labels.txt'
        
        # If labels file doesn't exist, download it
        if not os.path.exists(labels_path):
            print("Downloading ImageNet labels...")
            url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
            response = requests.get(url)
            labels = [line.strip() for line in response.text.splitlines()]
            
            # Save labels locally for future use
            with open(labels_path, 'w') as f:
                f.write('\n'.join(labels))
            
            return labels
        
        # If file exists, read it
        with tf.io.gfile.GFile(labels_path, 'r') as f:
            return [line.strip() for line in f]

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image for model input"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def detect_objects(self, image_path):  # This is the corrected method name
        """Detect objects in the image"""
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
            
            result = self.detector(converted_img)
            
            objects_found = []
            for i, score in enumerate(result['detection_scores'][0]):
                if score > 0.5:  # confidence threshold
                    class_name = result['detection_class_entities'][0][i].numpy().decode('utf-8')
                    confidence = float(score)
                    box = result['detection_boxes'][0][i].numpy()
                    objects_found.append({
                        'object': class_name,
                        'confidence': confidence,
                        'location': box.tolist()
                    })
            
            return objects_found
        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            return []

    def classify_image(self, image_path):
        """Classify the main subject of the image"""
        try:
            img = self.load_and_preprocess_image(image_path)
            predictions = self.classifier(img[tf.newaxis, ...])
            predicted_class = tf.argmax(predictions, axis=1)
            class_name = self.imagenet_labels[predicted_class[0]]
            confidence = float(tf.nn.softmax(predictions)[0][predicted_class[0]])
            
            return {
                'main_subject': class_name,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error in classify_image: {str(e)}")
            return {'main_subject': 'unknown', 'confidence': 0.0}

    def extract_features(self, image_path):
        """Extract high-level features from the image"""
        try:
            img = self.load_and_preprocess_image(image_path)
            features = self.feature_extractor(img[tf.newaxis, ...])
            return features[0].numpy()
        except Exception as e:
            print(f"Error in extract_features: {str(e)}")
            return np.zeros(1280)  # Return zero vector in case of error

    def analyze_image(self, image_path):
        """Perform comprehensive analysis of an image"""
        print(f"\nAnalyzing image: {image_path}")
        
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
            
            # Extract feature vector
            features = self.extract_features(image_path)
            
            analysis = {
                'image_info': image_info,
                'objects_detected': objects,
                'classification': classification,
                'feature_vector': features.tolist()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {str(e)}")
            return None

    def analyze_directory(self, directory='.', save_results=True):
        """Analyze all images in a directory"""
        results = {}
        
        # Get all image files
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_images = len(image_files)
        print(f"Found {total_images} images to analyze")
        
        # Add progress counter
        for idx, img_file in enumerate(image_files, 1):
            print(f"\nProcessing image {idx}/{total_images}: {img_file}")
            img_path = os.path.join(directory, img_file)
            try:
                results[img_file] = self.analyze_image(img_path)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                results[img_file] = None
            
            # Print progress percentage
            progress = (idx / total_images) * 100
            print(f"Overall progress: {progress:.1f}%")
        
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            results[img_file] = self.analyze_image(img_path)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'image_analysis_{timestamp}.json'
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nAnalysis results saved to {output_file}")
        
        return results

def main():
    print("Initializing Image Analyzer...")
    analyzer = ImageAnalyzer()
    print("Starting directory analysis...")
    results = analyzer.analyze_directory()
    
    # Print summary of findings
    print("\nAnalysis Summary:")
    for img_name, analysis in results.items():
        if analysis:
            print(f"\n{img_name}:")
            print(f"- Main subject: {analysis['classification']['main_subject']} "
                  f"(confidence: {analysis['classification']['confidence']:.2f})")
            print("- Objects detected:")
            for obj in analysis['objects_detected']:
                print(f"  * {obj['object']} (confidence: {obj['confidence']:.2f})")

if __name__ == "__main__":
    main()
