import os
import subprocess
import glob


def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run ffmpeg command to extract frames
    command = f'ffmpeg -i "{video_path}" -vf fps=1 "{output_dir}/%d.jpg"'
    subprocess.call(command, shell=True)


def main(input_dir, output_dir):
    # Get list of mp4 files in input directory
    mp4_files = glob.glob(os.path.join(input_dir, '*.mp4'))

    for mp4_file in mp4_files:
        filename = os.path.splitext(os.path.basename(mp4_file))[0]
        output_subdir = os.path.join(output_dir, filename)
        extract_frames(mp4_file, output_subdir)


if __name__ == "__main__":
    input_directory = '/path/to/input/directory'
    output_directory = '/path/to/output/directory'
    main(input_directory, output_directory)
