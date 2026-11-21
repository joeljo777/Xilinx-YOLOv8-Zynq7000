"""
Fast COCO Dataset Download for Vitis-AI Quantization
Downloads validation images + annotations
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    print("=" * 70)
    print("COCO 2017 VALIDATION SET DOWNLOAD")
    print("=" * 70)
    
    # Create directories
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(exist_ok=True)
    
    # COCO URLs
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    images_zip = dataset_dir / "val2017.zip"
    annotations_zip = dataset_dir / "annotations_trainval2017.zip"
    
    # Download validation images (~1GB)
    if not (dataset_dir / "val2017").exists():
        print("\n[1/2] Downloading validation images (~1GB)...")
        print("This may take 10-30 minutes depending on your connection\n")
        download_url(images_url, str(images_zip))
        
        print("\nExtracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Clean up zip
        images_zip.unlink()
        print(f"✓ Images extracted to: {dataset_dir / 'val2017'}")
    else:
        print(f"\n✓ Validation images already exist: {dataset_dir / 'val2017'}")
    
    # Download annotations (~241MB)
    if not (dataset_dir / "annotations").exists():
        print("\n[2/2] Downloading annotations (~241MB)...\n")
        download_url(annotations_url, str(annotations_zip))
        
        print("\nExtracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Clean up zip
        annotations_zip.unlink()
        print(f"✓ Annotations extracted to: {dataset_dir / 'annotations'}")
    else:
        print(f"\n✓ Annotations already exist: {dataset_dir / 'annotations'}")
    
    # Verify
    val_images = list((dataset_dir / "val2017").glob("*.jpg"))
    print("\n" + "=" * 70)
    print(f"✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"Images: {len(val_images)} JPG files")
    print(f"Location: {dataset_dir.absolute()}")
    print("\n📊 Next step: Run accuracy measurement")
    print("   python measure_accuracy.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
