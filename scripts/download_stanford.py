import os
import wget
import tarfile
from pathlib import Path

def download_and_extract_dataset():
    # Create directories if they don't exist
    base_dir = Path("data")
    raw_dir = base_dir / "raw" / "stanford_dogs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # URLs for the dataset files
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

    # Download files
    for url in [images_url, annotations_url, lists_url]:
        filename = raw_dir / Path(url).name
        if not filename.exists():
            print(f"Downloading {url}...")
            wget.download(url, str(filename))
            print("\nDownload complete!")

        # Extract files
        print(f"Extracting {filename}...")
        with tarfile.open(filename) as tar:
            tar.extractall(path=raw_dir)
        print("Extraction complete!")

if __name__ == "__main__":
    print("Starting Stanford Dogs Dataset download...")
    download_and_extract_dataset()
    print("Dataset download and extraction completed!")