import os
from PIL import Image
from tqdm import tqdm
import concurrent.futures


DATA_DIR = "./data"
# This script will verify every image in your dataset and remove corrupted ones.

def verify_image(file_path):
    """
    Tries to open and fully load an image file. 
    If any part of this process fails, it returns the path for deletion.
    Returns None if the image is valid.
    """
    try:
        # Open the image file
        img = Image.open(file_path)
        # The key is to fully load the image data into memory.
        # This will trigger an error for a wide range of corruptions.
        # We also check that it's a recognized format.
        img.load()
        
        # A final check to ensure it's a format we can use (e.g., not a weird BMP variant)
        if img.format.lower() not in ['jpeg', 'png', 'bmp']:
             print(f"Unsupported format found: {img.format} at {file_path}")
             return file_path

        return None # If all checks pass, the image is good.
    except Exception as e:
        # If ANY exception occurs (IOError, UnidentifiedImageError, the NoneType error, etc.),
        # we know the file is bad.
        print(f"Corrupted file detected: {file_path} (Reason: {e})")
        return file_path

def clean_dataset_parallel(root_dir):
    """
    Uses multiple CPU cores to speed up the image verification process.
    """
    print(f"--- Starting Parallel Dataset Cleaning in: {root_dir} ---")
    
    # 1. Find all image file paths
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(image_paths)} total images to verify.")
    
    # 2. Use a process pool to check images in parallel
    corrupted_files = []
    # Use tqdm to show a progress bar
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map the verify_image function to all paths
        results = list(tqdm(executor.map(verify_image, image_paths), total=len(image_paths), desc="Verifying Images"))

    # 3. Filter out the None results to get the list of corrupted files
    corrupted_files = [path for path in results if path is not None]
    
    # 4. Delete the corrupted files
    if not corrupted_files:
        print("\n--- Dataset cleaning complete. No corrupted files were found! ---")
    else:
        print(f"\n--- Found {len(corrupted_files)} corrupted files. Deleting them now... ---")
        for file_path in corrupted_files:
            try:
                os.remove(file_path)
                print(f"DELETED: {file_path}")
            except OSError as e:
                print(f"ERROR deleting {file_path}: {e}")
        print(f"\n--- Dataset cleaning complete. Removed {len(corrupted_files)} files. ---")

# --- RUN THE CLEANING PROCESS ---
# This will check your train, val, and test sets.
clean_dataset_parallel(DATA_DIR)