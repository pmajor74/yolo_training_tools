import sys
import os
from pathlib import Path

def clear_console():
    """Clears the console screen based on the operating system."""
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')

# clear the console early so its easy to track output and errors
clear_console()

from yolo_viewer.main_application import main

if __name__ == "__main__":
    main()