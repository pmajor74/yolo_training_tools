"""Convert 1-bit TIF files to RGB format for YOLO training."""

from pathlib import Path
from PIL import Image
import shutil
from typing import List

def convert_tif_to_rgb(input_path: Path, output_path: Path = None) -> bool:
    """
    Convert a TIF file to RGB format if needed.
    
    Args:
        input_path: Path to input TIF file
        output_path: Path to output file (if None, overwrites input)
    
    Returns:
        True if conversion was needed and successful, False otherwise
    """
    try:
        # Open the image
        img = Image.open(input_path)
        
        # Check if conversion is needed
        if img.mode != 'RGB':
            print(f"Converting {input_path.name} from mode {img.mode} to RGB")
            
            # Convert to RGB
            if img.mode == '1':  # 1-bit black and white
                # Convert to L (grayscale) first, then to RGB
                rgb_img = img.convert('L').convert('RGB')
            elif img.mode == 'L':  # Grayscale
                rgb_img = img.convert('RGB')
            elif img.mode == 'RGBA':  # Has alpha channel
                # Create white background and paste image
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
            else:
                # Generic conversion
                rgb_img = img.convert('RGB')
            
            # Save the converted image with LZW compression (works with RGB)
            save_path = output_path or input_path
            rgb_img.save(save_path, 'TIFF', compression='lzw')
            print(f"  Saved to {save_path}")
            return True
        else:
            print(f"Skipping {input_path.name} - already RGB")
            return False
            
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def convert_dataset_tifs(dataset_path: Path, backup: bool = True) -> None:
    """
    Convert all TIF files in a dataset to RGB format.
    
    Args:
        dataset_path: Path to dataset root (containing train/val/test folders)
        backup: Whether to backup original files
    """
    # Find all TIF files
    tif_files = list(dataset_path.glob('**/*.tif')) + list(dataset_path.glob('**/*.tiff'))
    
    if not tif_files:
        print(f"No TIF files found in {dataset_path}")
        return
    
    print(f"Found {len(tif_files)} TIF files to process")
    
    # Create backup directory if requested
    if backup:
        backup_dir = dataset_path.parent / f"{dataset_path.name}_tif_backup"
        if not backup_dir.exists():
            backup_dir.mkdir(parents=True)
            print(f"Created backup directory: {backup_dir}")
    
    # Process each file
    converted_count = 0
    for tif_path in tif_files:
        # Create backup if requested
        if backup:
            rel_path = tif_path.relative_to(dataset_path)
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(tif_path, backup_path)
        
        # Convert the file in place
        if convert_tif_to_rgb(tif_path):
            converted_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Converted: {converted_count} files")
    print(f"  Already RGB: {len(tif_files) - converted_count} files")
    if backup:
        print(f"  Originals backed up to: {backup_dir}")


def main():
    """Main function to convert TIF files in the working directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TIF files to RGB format for YOLO training')
    parser.add_argument('path', nargs='?', default='working', 
                        help='Path to dataset directory (default: working)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not backup original files')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.path)
    
    if not dataset_path.exists():
        print(f"Error: Path {dataset_path} does not exist")
        return
    
    # Check if it looks like a dataset directory
    has_splits = any((dataset_path / split).exists() for split in ['train', 'val', 'test'])
    
    if not has_splits:
        response = input(f"Warning: {dataset_path} doesn't have train/val/test folders. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run conversion
    convert_dataset_tifs(dataset_path, backup=not args.no_backup)


if __name__ == '__main__':
    main()