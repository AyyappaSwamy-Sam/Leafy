# NEW CELL: DATASET CLEANING SCRIPT

import os
from PIL import Image
from tqdm import tqdm
DATA_DIR = "./new_data"
def clean_dataset(base_dir):
    """
    Iterates through all files in a directory, tries to open them as images,
    and deletes them if they are corrupt.
    """
    print(f"Starting dataset cleaning in: {base_dir}")
    corrupted_files_found = 0
    
    # Use os.walk to go through all subdirectories
    for dirpath, _, filenames in tqdm(os.walk(base_dir), desc="Scanning directories"):
        for filename in filenames:
            # We only care about common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(dirpath, filename)
                try:
                    # Open the image file
                    img = Image.open(file_path)
                    # This is a more thorough check that the file can be read
                    img.verify() 
                except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
                    print(f"\n---> Corrupted file found: {file_path}")
                    print(f"     Reason: {e}")
                    # Delete the corrupted file
                    os.remove(file_path)
                    print(f"     DELETED.")
                    corrupted_files_found += 1
    
    if corrupted_files_found == 0:
        print("\nDataset cleaning complete. No corrupted files were found.")
    else:
        print(f"\nDataset cleaning complete. Total corrupted files removed: {corrupted_files_found}")

# --- RUN THE CLEANING PROCESS ---
# This will check your train, val, and test sets.
clean_dataset(DATA_DIR)