from pathlib import Path
from fingerprint_pipeline import FingerprintPipeline

def main():
    # Configuration
    INPUT_FOLDER = "../fingerprints/"  # Folder containing fingerprint images
    OUTPUT_FOLDER = "../results"      # Where to save results
    
    # Create pipeline
    pipeline = FingerprintPipeline()
    
    # Get all fingerprint images
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"❌ Error: Folder '{INPUT_FOLDER}' not found!")
        print(f"Please create a '{INPUT_FOLDER}' folder and add fingerprint images to it.")
        print("\nSupported formats: .tif, .jpg, .jpeg, .png, .bmp")
        return
    
    # Find all images
    image_files = []
    for extension in ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp']:
        # rglob handles recursive searching through all subdirectories
        image_files.extend(input_path.rglob(extension))
        image_files.extend(input_path.rglob(extension.upper()))
    
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if not image_files:
        print(f"❌ No images found in '{INPUT_FOLDER}' folder!")
        print("Please add fingerprint images with these extensions: .tif, .jpg, .png, .bmp")
        return
    
    print("="*70)
    print("FINGERPRINT BATCH PROCESSING")
    print("="*70)
    print(f"Found {len(image_files)} image(s)\n")
    
    # Process each image
    results_summary = []
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            # Create output directory for this image
            image_output = output_path / image_path.stem
            
            # Run pipeline
            minutiae = pipeline.run(str(image_path), str(image_output))
            
            # Calculate statistics
            endings = sum(1 for m in minutiae if m.type == 'Ending')
            bifurcations = sum(1 for m in minutiae if m.type == 'Bifurcation')
            
            # Display results
            print(f"  ✓ Success!")
            print(f"    Total minutiae: {len(minutiae)}")
            print(f"    Endings: {endings}")
            print(f"    Bifurcations: {bifurcations}")
            print(f"    Output: {image_output}\n")
            
            # Store result
            results_summary.append({
                'filename': image_path.name,
                'status': 'success',
                'total_minutiae': len(minutiae),
                'endings': endings,
                'bifurcations': bifurcations
            })
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
            
            results_summary.append({
                'filename': image_path.name,
                'status': 'failed',
                'error': str(e)
            })
            error_count += 1
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total processed: {len(image_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    
    if success_count > 0:
        total_minutiae = sum(r['total_minutiae'] for r in results_summary if r['status'] == 'success')
        avg_minutiae = total_minutiae / success_count
        print(f"Total minutiae extracted: {total_minutiae}")
        print(f"Average per image: {avg_minutiae:.1f}")
    
    print(f"\nResults saved to: {output_path}/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
