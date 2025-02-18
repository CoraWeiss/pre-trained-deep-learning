import tensorflow as tf
import tensorflow_hub as hub
import os
import csv
from datetime import datetime
import requests

class ImageClassifier:
    def __init__(self):
        print("Loading model...")
        self.model = hub.load('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2')
        self.labels = self._get_labels()

    def _get_labels(self):
        """Load or download ImageNet labels"""
        labels_path = 'imagenet_labels.txt'
        
        # Create labels file if it doesn't exist
        if not os.path.exists(labels_path):
            print("Downloading labels...")
            url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
            response = requests.get(url)
            labels = [line.strip() for line in response.text.splitlines()]
            
            # Save labels locally
            with open(labels_path, 'w') as f:
                f.write('\n'.join(labels))
            return labels
        
        # Read existing labels file
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f]

    def classify_image(self, image_path):
        """Classify a single image"""
        try:
            # Load and preprocess image
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224))
            img = tf.cast(img, tf.float32) / 255.0
            
            # Get prediction
            predictions = self.model(img[tf.newaxis, ...])
            predicted_class = tf.argmax(predictions, axis=1)
            confidence = float(tf.nn.softmax(predictions)[0][predicted_class[0]])
            
            return {
                'image': os.path.basename(image_path),
                'subject': self.labels[predicted_class[0]],
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self, directory='.'):
        """Process all images in a directory"""
        # Get all images
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not image_files:
            print("No images found in directory")
            return
        
        # Process images and save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'classifications_{timestamp}.csv'
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'subject', 'confidence'])
            writer.writeheader()
            
            total = len(image_files)
            for idx, img_file in enumerate(image_files, 1):
                img_path = os.path.join(directory, img_file)
                result = self.classify_image(img_path)
                
                if result:
                    writer.writerow(result)
                
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        
        print(f"\nResults saved to {output_file}")

def main():
    # Initialize classifier
    classifier = ImageClassifier()
    
    # Process current directory
    classifier.process_directory()

if __name__ == "__main__":
    main()
