import cv2
import os

def extract_frames(video_path, output_folder, num_frames=24):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Capture the video from the given path
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames to extract
    interval = max(total_frames // num_frames, 1)
    
    frame_count = 0
    extracted_frame_count = 0

    while video_capture.isOpened() and extracted_frame_count < num_frames:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Save the frame if it matches the interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {extracted_frame_count} frames to {output_folder}")

# Usage
video_path = r"D:\helmet\video\1.mp4"
output_folder = r"D:\helmet\Photos"
extract_frames(video_path, output_folder)
