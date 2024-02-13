import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch


def extract_faces_from_jpg_folder(input_folder, output_folder, mtcnn):
    # Get a list of all jpg files in the input folder
    jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for jpg_file in jpg_files:
        jpg_path = os.path.join(input_folder, jpg_file)
        image = Image.open(jpg_path)

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(image)

        if boxes is None:
            print(f"No faces detected in {jpg_file}")
            continue

        for i, box in enumerate(boxes):
            # Convert box coordinates to integers
            box = box.astype(int)
            # Crop face from image
            cropped_face = image.crop((box[0], box[1], box[2], box[3]))
            # Save cropped face as new image
            output_path = os.path.join(output_folder, f"{jpg_file[:-4]}_face_{i}.jpg")
            cropped_face.save(output_path)
            print(f"Face {i + 1} from {jpg_file} saved to {output_path}")


def load_mtcnn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,  # do not perform tensor normalization
        keep_all=True,
        device=device
    )
    return mtcnn


if __name__ == "__main__":
    input_folder = '/path/to/input/folder'
    output_folder = '/path/to/output/folder'

    # Initialize MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = load_mtcnn()

    # Extract faces from jpg files in input folder
    extract_faces_from_jpg_folder(input_folder, output_folder, mtcnn)
