import os
import zipfile
import urllib.request

ROBOT_URL = "https://www.dropbox.com/scl/fi/uewvrcempf2wf2jp7bcb8/robot.zip?rlkey=7uwz1ne94hxyinub8x16y93em&dl=1"


def ensure_robot_parts(robot_dir="robot"):
    if all(os.path.exists(os.path.join(robot_dir, name)) for name in ["Base.stl", "BaseRot.stl", "Humerus.stl", "Radius.stl"]):
        print("Robot STL files already present.")
        return robot_dir

    os.makedirs(robot_dir, exist_ok=True)
    zip_path = "robot.zip"
    print("Downloading robot STL archive...")
    urllib.request.urlretrieve(ROBOT_URL, zip_path)
    print("Extracting robot STL archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    os.remove(zip_path)
    print("Robot STL files ready.")
    return robot_dir


if __name__ == "__main__":
    ensure_robot_parts()
