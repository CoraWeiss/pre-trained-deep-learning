import json
import os

# Find the most recent analysis file
analysis_files = [f for f in os.listdir('.') if f.startswith('image_analysis_') and f.endswith('.json')]
latest_file = max(analysis_files, key=os.path.getctime) if analysis_files else None

if latest_file:
    # Load the results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Print basic stats
    print(f"Analysis file: {latest_file}")
    print(f"Number of images analyzed: {len(results)}")
    
    # Show a sample result (first image)
    sample_image = next(iter(results))
    print(f"\nSample analysis for: {sample_image}")
    print(json.dumps(results[sample_image], indent=2))
    
    # Display top 5 classifications by confidence
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
    
    print("\nTop 5 most confident classifications:")
    for i, item in enumerate(classifications[:5], 1):
        print(f"{i}. {item['image']}: {item['subject']} (confidence: {item['confidence']:.4f})")
else:
    print("No analysis files found in current directory.")
