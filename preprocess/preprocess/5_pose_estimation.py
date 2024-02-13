import cv2
from headpose_estimation import Headpose
import os
import glob
from shutil import copyfile
import argparse

headpose = Headpose()


def fun(img_path):
    # img = cv2.imread("tmp2.jpg")
    img = cv2.imread(img_path)
    detections, image = headpose.run(img)
    dic = detections[0]
    # {'bbox': array([228, 139, 530, 571]),
    #   'yaw': 17.79396,
    #   'pitch': -12.596962,
    #   'roll': 2.096115}
    # print(dic['yaw'], dic['pitch'], dic['roll'])
    return dic


def copy_images_with_small_poses(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of jpg files in input folder
    jpg_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for jpg_file in jpg_files:
        # Perform pose estimation
        pose = fun(jpg_file)

        # Check if yaw, pitch, and roll are all within the threshold
        if abs(pose['yaw']) < 25 and abs(pose['pitch']) < 25 and abs(pose['roll']) < 25:
            # Copy the image to the output folder
            filename = os.path.basename(jpg_file)
            output_path = os.path.join(output_folder, filename)
            copyfile(jpg_file, output_path)
            print(f"Image {filename} copied to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images with small poses.")
    parser.add_argument("input_folder", type=str, help="Path to input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to output folder for copied images.")
    args = parser.parse_args()
    copy_images_with_small_poses(args.input_folder, args.output_folder)
