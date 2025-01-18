import subprocess 
import os
import shutil


def delete_directory_if_exists(path:str):
    if os.path.exists(path) and os.path.isdir(path):
        print("Removing directory: ", path)
        try:
            os.rmdir(path)
        except OSError:
            shutil.rmtree(path)

def delete_file_if_exists(path:str):
    if os.path.exists(path) and os.path.isfile(path):
        print("Removing file: ", path)
        try:
            os.remove(path)
        except OSError:
            os.system(f"rm {path}")
    
def move_file_here(path:str):
    if os.path.exists(path) and os.path.isfile(path):
        print("Moving file: ", path)
        try:
            shutil.move(path, ".")
        except OSError:
            os.system(f"mv {path} .")


if __name__ == "__main__":


    print("Building combine...")

    proc = subprocess.run(
        [
            "pyinstaller", "--onefile", "main.py", "--name", "combine"
        ],
        capture_output=True, 
        text=True,
        check=True)

    print("Done.")
    print("Moving combine.exe to root...")
    move_file_here("dist/combine.exe")
    print("Cleaning up...")
    delete_directory_if_exists("build")
    delete_directory_if_exists("dist")
    delete_file_if_exists("combine.spec")
    print("Done.")

