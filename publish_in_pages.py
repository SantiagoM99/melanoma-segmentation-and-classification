import shutil
import os

def transfer_files(source_dir, destination_dir):
    """
    Transfers all files and folders from source_dir to destination_dir.

    Parameters:
    - source_dir (str): Path to the source directory.
    - destination_dir (str): Path to the destination directory.
    """
    # Check if source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Move all files and folders from source_dir to destination_dir
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        dest_path = os.path.join(destination_dir, item)

        # Remove existing destination files/folders
        if os.path.exists(dest_path):
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)
            else:
                os.remove(dest_path)

        # Move files or folders
        shutil.move(source_path, dest_path)

    print(f"All files from '{source_dir}' have been moved to '{destination_dir}'.")

# Example usage
if __name__ == "__main__":
    transfer_files("docs/build/html/", "docs/")
