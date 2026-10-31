"""Download Flickr8k with KaggleHub and print the project's training command."""

from pathlib import Path

import kagglehub


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "flickr8k"
DATASET_ID = "adityajn105/flickr8k"


def main():
    dataset_root = Path(
        kagglehub.dataset_download(DATASET_ID, output_dir=str(OUTPUT_DIR))
    )
    image_dirs = [
        path for path in dataset_root.rglob("Images")
        if path.is_dir() and any(path.glob("*.jpg"))
    ]
    caption_files = [
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.name.lower() in {
            "captions.txt", "flickr8k.token.txt", "flickr8k.token"
        }
    ]

    print(f"Dataset location: {dataset_root}")
    if not image_dirs or not caption_files:
        print("Could not automatically find the Images folder and captions file.")
        print("Files found:")
        for path in dataset_root.rglob("*"):
            if path.is_file():
                print(f"  {path.relative_to(dataset_root)}")
        return

    images_dir = image_dirs[0]
    captions_file = caption_files[0]
    image_count = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"Images folder: {images_dir} ({image_count} JPG images)")
    print(f"Captions file: {captions_file}")
    print("\nRun this next from the project folder:")
    print(
        f'python train.py --images "{images_dir}" '
        f'--captions "{captions_file}" --epochs 10'
    )


if __name__ == "__main__":
    main()
