import cv2
import threading 
from ultralytics import YOLO 

# Object Tracking Code for Multistreaming
# Define the video files for the trackers as file1 for input video in .mp4 and file2 for WebCam path
video_file1 = 'videos/COD_MW_3.mp4'
video_file2 = 0 

# load YOLOv8 models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8s.pt')

# target function for the thread
'''
This target will be a function responsible for running the video file and applying the YOLOv8 object-tracking algorithm. The mentioned code will act as the target function for our thread.
'''
def run_tracker_in_thread(filename, model, file_index):
    """
    This function is designed to run a video file or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - filename: The path to the video file or the webcam/external
    camera source.

    - model: The file path to the YOLOv8 model.

    - file_index: An argument to specify the count of the
    file being processed.
    """
    video = cv2.VideoCapture(filename)
    while True:
        # read the video frames
        ret, frame = video.read()
        # exit the loop if no more frames in either video
        if not ret:
            break
        # track the objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.imshow("Tracking_Stream_"+str(file_index), res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # release video sources
    video.release()

'''
The file_index parameter represents the index of the file that can be assigned to each thread. For example, if you are processing two videos concurrently, the first thread will be assigned file_index as 1, while the second thread will use file_index as 2. In the case of a third thread, it will have file_index set to 3.
'''
'''
Hint: If you want to do Object Detection on Multiple streams, you can replace model.track with model.predict in the above function.
'''

# create the object tracking threads
'''
Generating the threads for object tracking. The number of threads will correspond to the number of video sources you intend to run. The provided code will create two threads to handle two video streams.
'''
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file1, model1, 1),
                                   daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file2, model2, 2),
                                   daemon=True)

# start the objects tracking threads
tracker_thread1.start()
tracker_thread2.start()

# Thread handling and destroy windows
tracker_thread1.join()
tracker_thread2.join()

# clean up and close windows
cv2.destroyAllWindows()