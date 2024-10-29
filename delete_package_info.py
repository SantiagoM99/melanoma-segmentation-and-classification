"""
This script deletes the package folders of Build and Dist made when creating the package.
It also deletes the .egg-info folder created when installing the package.

This script is useful when you want to delete the package folders to avoid confusion when creating a new package.
Also because, twine will not allow you to upload a package with the same version number and when there's more
than one package folder it will try to upload all of them.
"""
import os
import shutil

def delete_folder(folder_name: str) -> None:
    """
    Deletes the folder with the given name.

    Parameters
    ----------
    folder_name : str
        Name of the folder to be deleted.

    Returns
    -------
    None
    """
    #Get current working directory
    cwd = os.getcwd()
    #Join the current working directory with the folder name
    folder = os.path.join(cwd, folder_name)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder_name}")
    else:
        print(f"{folder_name} does not exist")


def delete_package_info() -> None:
    """
    Deletes the package folders of Build and Dist and the .egg-info folder.

    Returns
    -------
    None
    """
    delete_folder("build")
    delete_folder("dist")
    delete_folder("melanoma_segmentation.egg-info")


if __name__ == "__main__":
    delete_package_info()