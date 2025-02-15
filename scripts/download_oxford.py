import os
import wget
import tarfile
from pathlib import Path


def download_and_extract_dataset():
    # Create directories if they don't exist
    base_dir = Path("data")
    raw_dir = base_dir / "raw" / "oxford_pets"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Dataset URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = (
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    )

    # Download and extract each file
    for url in [images_url, annotations_url]:
        filename = raw_dir / Path(url).name
        if not filename.exists():
            print(f"Downloading {url}...")
            wget.download(url, str(filename))
            print("\nDownload complete!")

        # Extract files
        print(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted path traversal in tar file")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, path=raw_dir)
        print("Extraction complete!")

    print("\nVerifying dataset structure...")
    # Verify the extracted files
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"

    if images_dir.exists() and annotations_dir.exists():
        image_count = len(list(images_dir.glob("*.jpg")))
        print(f"Found {image_count} images")
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Error: Dataset extraction incomplete!")


if __name__ == "__main__":
    print("Starting Oxford-IIIT Pet Dataset download...")
    download_and_extract_dataset()
