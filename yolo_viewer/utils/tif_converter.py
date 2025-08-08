"""Utility for checking and converting TIF files to RGB format for YOLO training."""

from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import shutil

from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt6.QtCore import Qt


class TifFormatChecker:
    """Check and convert TIF files to RGB format."""
    
    @staticmethod
    def check_tif_files(dataset_path: Path) -> Tuple[List[Path], List[Path]]:
        """
        Check TIF files in dataset for non-RGB format.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (non_rgb_files, all_tif_files)
        """
        # Find all TIF files
        tif_files = []
        for pattern in ['**/*.tif', '**/*.tiff']:
            tif_files.extend(list(dataset_path.glob(pattern)))
        
        if not tif_files:
            return [], []
        
        non_rgb_files = []
        for tif_path in tif_files:
            try:
                with Image.open(tif_path) as img:
                    if img.mode != 'RGB':
                        non_rgb_files.append(tif_path)
            except Exception:
                # If we can't read the file, consider it problematic
                non_rgb_files.append(tif_path)
        
        return non_rgb_files, tif_files
    
    @staticmethod
    def show_conversion_dialog(parent, non_rgb_count: int, total_tif_count: int) -> bool:
        """
        Show dialog asking user if they want to convert TIF files.
        
        Args:
            parent: Parent widget for dialog
            non_rgb_count: Number of non-RGB TIF files
            total_tif_count: Total number of TIF files
            
        Returns:
            True if user wants to convert, False otherwise
        """
        message = f"""
<b>TIF File Format Issue Detected</b>

Found {non_rgb_count} TIF files that are not in RGB format (out of {total_tif_count} total TIF files).

YOLO training requires RGB images with 3 channels, but some of your TIF files are:
• 1-bit black and white images
• Grayscale images  
• Other non-RGB formats

<b>Would you like to convert these files to RGB format?</b>

<i>What will happen:</i>
• Original files will be backed up with .bak extension
• Files will be converted to RGB format with same names
• Your annotation files (.txt) will remain unchanged
• Training can then proceed normally

<i>If you choose "No", training will be cancelled.</i>
        """
        
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle("Convert TIF Files?")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(message.strip())
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        msg_box.setIcon(QMessageBox.Icon.Question)
        
        result = msg_box.exec()
        return result == QMessageBox.StandardButton.Yes
    
    @staticmethod
    def convert_tif_files(non_rgb_files: List[Path], parent=None, log_func=None) -> bool:
        """
        Convert TIF files to RGB format with progress dialog.
        
        Args:
            non_rgb_files: List of TIF files to convert
            parent: Parent widget for progress dialog
            log_func: Optional logging function
            
        Returns:
            True if all conversions successful, False otherwise
        """
        def log(message):
            if log_func:
                log_func(message)
            else:
                print(message)
                
        if not non_rgb_files:
            return True
        
        # Create progress dialog
        progress = QProgressDialog(f"Converting {len(non_rgb_files)} TIF files to RGB format...", 
                                 "Cancel", 0, len(non_rgb_files), parent)
        progress.setWindowTitle("Converting TIF Files")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        success_count = 0
        failed_files = []
        
        for i, tif_path in enumerate(non_rgb_files):
            if progress.wasCanceled():
                break
                
            progress.setLabelText(f"Converting: {tif_path.name}")
            progress.setValue(i)
            QApplication.processEvents()
            
            try:
                # Create backup
                backup_path = tif_path.with_suffix(tif_path.suffix + '.bak')
                if not backup_path.exists():
                    shutil.copy2(tif_path, backup_path)
                
                # Convert image
                with Image.open(tif_path) as img:
                    if img.mode != 'RGB':
                        # Convert to RGB
                        if img.mode == '1':  # 1-bit black and white
                            rgb_img = img.convert('L').convert('RGB')
                        elif img.mode == 'L':  # Grayscale
                            rgb_img = img.convert('RGB')
                        elif img.mode == 'RGBA':  # Has alpha channel
                            # Create white background and paste image
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if len(img.split()) > 3:
                                rgb_img.paste(img, mask=img.split()[3])
                            else:
                                rgb_img.paste(img)
                        else:
                            # Generic conversion
                            rgb_img = img.convert('RGB')
                        
                        # Save converted image with LZW compression
                        rgb_img.save(tif_path, 'TIFF', compression='lzw')
                        success_count += 1
                    else:
                        success_count += 1  # Already RGB
                        
            except Exception as e:
                failed_files.append((tif_path, str(e)))
        
        progress.setValue(len(non_rgb_files))
        progress.close()
        
        # Show results
        if failed_files:
            failed_names = [f.name for f, _ in failed_files]
            QMessageBox.warning(
                parent,
                "Conversion Issues",
                f"Successfully converted {success_count}/{len(non_rgb_files)} files.\n\n"
                f"Failed to convert {len(failed_files)} files:\n" + 
                "\n".join(failed_names[:5]) + 
                ("..." if len(failed_names) > 5 else "")
            )
            return False
        else:
            QMessageBox.information(
                parent,
                "Conversion Complete",
                f"Successfully converted all {success_count} TIF files to RGB format.\n\n"
                f"Original files backed up with .bak extension."
            )
            return True
    
    @staticmethod
    def check_and_convert_if_needed(dataset_path: Path, parent=None, log_func=None) -> bool:
        """
        Check TIF files and prompt for conversion if needed.
        
        Args:
            dataset_path: Path to dataset directory
            parent: Parent widget for dialogs
            log_func: Optional logging function
            
        Returns:
            True if ready for training (no issues or conversion successful), False otherwise
        """
        def log(message):
            print(f"TIF_CONVERTER: {message}")  # Always print to console
            if log_func:
                log_func(message)
        
        print(f"TIF_CONVERTER: Starting TIF check in dataset: {dataset_path}")
        print(f"TIF_CONVERTER: Dataset exists: {dataset_path.exists()}")
        
        non_rgb_files, all_tif_files = TifFormatChecker.check_tif_files(dataset_path)
        
        print(f"TIF_CONVERTER: Found {len(all_tif_files)} total TIF files")
        print(f"TIF_CONVERTER: Found {len(non_rgb_files)} non-RGB TIF files")
        
        if all_tif_files:
            log(f"Found {len(all_tif_files)} TIF files in dataset")
            # List the files for debugging
            for i, tif_file in enumerate(all_tif_files[:5]):  # Show first 5
                print(f"TIF_CONVERTER: TIF file {i+1}: {tif_file}")
            if len(all_tif_files) > 5:
                print(f"TIF_CONVERTER: ... and {len(all_tif_files) - 5} more")
        
        if non_rgb_files:
            log(f"Detected {len(non_rgb_files)} TIF files that need RGB conversion")
            # List the problematic files
            for i, tif_file in enumerate(non_rgb_files[:3]):  # Show first 3
                print(f"TIF_CONVERTER: Non-RGB file {i+1}: {tif_file}")
        
        if not non_rgb_files:
            # No issues found
            if all_tif_files:
                log(f"All {len(all_tif_files)} TIF files are already in RGB format")
            else:
                print("TIF_CONVERTER: No TIF files found in dataset")
            return True
        
        print(f"TIF_CONVERTER: Showing conversion dialog to user...")
        
        # Ask user if they want to convert
        if TifFormatChecker.show_conversion_dialog(parent, len(non_rgb_files), len(all_tif_files)):
            # User wants to convert
            log(f"Converting {len(non_rgb_files)} TIF files to RGB format...")
            result = TifFormatChecker.convert_tif_files(non_rgb_files, parent, log_func)
            if result:
                log("TIF file conversion completed successfully")
            else:
                log("TIF file conversion failed")
            return result
        else:
            # User declined conversion
            log("TIF file conversion cancelled by user")
            return False