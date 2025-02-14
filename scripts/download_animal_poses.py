import os
import sys
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_and_organize_dataset():
    # Create directories
    base_dir = Path("data")
    raw_dir = base_dir / "raw" / "animal_poses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Expected files from the Animal Pose website
    expected_files = [
        "bndbox_anno.tar.gz",
        "bndbox_image.tar.gz", 
        "keypoints.json",
        "Self_collected_Images.tar.gz"
    ]

    # Check for downloaded files
    found_files = []
    for filename in expected_files:
        matches = list(raw_dir.glob(f"*{filename}"))  # More flexible matching
        if matches:
            found_files.append(matches[0])

    if not found_files:
        print("\nNo expected files found in:", raw_dir)
        print("Please download the files from https://sites.google.com/view/animal-pose/")
        print("and place them in:", raw_dir)
        print("\n".join(f"- {f}" for f in expected_files))
        sys.exit(1)

    # Extract all archive files found
    for file_path in found_files:
        if file_path.name.endswith('.json'):
            continue  # Skip json files as they don't need extraction
            
        print(f"\nExtracting {file_path.name}...")
        try:
            if file_path.name.endswith('.tar.gz'):
                with tarfile.open(file_path, 'r:gz') as tar:
                    # List all files to extract for progress bar
                    members = tar.getmembers()
                    for member in tqdm(members, desc=f"Extracting {file_path.name}"):
                        tar.extract(member, raw_dir)
            print(f"Successfully extracted {file_path.name}")
        except Exception as e:
            print(f"Error extracting {file_path.name}: {str(e)}")
            continue

    print("\nVerifying dataset structure...")
    # Basic verification - adjust these based on the actual structure
    expected_items = [
        "bndbox_anno",
        "bndbox_image",
        "keypoints.json",
        "Self_collected_Images"
    ]
    missing_items = [item for item in expected_items 
                    if not any((raw_dir / item).exists() for ext in ['', '.tar.gz', '.json'])]
    
    if missing_items:
        print(f"Warning: Missing items: {missing_items}")
        print("The dataset might not have been downloaded or extracted correctly.")
        print("\nExpected structure:")
        print(f"data/raw/animal_poses/")
        print("├── bndbox_anno/")
        print("├── bndbox_image/")
        print("├── keypoints.json")
        print("└── Self_collected_Images/")
    else:
        print("Dataset structure verified!")
        # Count images if possible
        try:
            image_count = len(list((raw_dir / "bndbox_image").glob("*.jpg")))
            print(f"Found {image_count} images in bndbox_image/")
        except Exception:
            print("Could not count images - directory structure might be different")

    # Cleanup suggestion
    print("\nNote: You can now safely delete the archive files to save space.")
    print("The following files can be deleted:")
    for file_path in found_files:
        if file_path.suffix in ['.gz']:
            print(f"- {file_path}")

if __name__ == "__main__":
    print("Starting Animal Pose Dataset organization...")
    extract_and_organize_dataset()