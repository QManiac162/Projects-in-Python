# for machine learning
from typing import Any
import torch 


# for array computation
import numpy as np


# for image decoding / editing
import cv2


# for environment variables
import os 


# for detecting which ML devices we can use
import platform


# for actually using the YOLO models
from ultralytics import YOLO


# constructor
'''
declaring a new class called YoloV8ImageObjectDetection and we have two class variables:

PATH - This will be the path to a pretrained model (checkpoint file). If you use the default, we will just load the pretrained one from torchhub

CONF_THRESH - This will be our confidence threshold for detections. For example, if our confidence is 0.5, it means that our model will only show and annotate detections in an image that have a 50% or higher confidence. Anything lower will be ignored.

Our __init__ takes a single parameter, chunked, which is the raw binary image stream that was delivered to the API. It then:

-> loads the model using the _load_model fuction
-> set's the most efficient device for our system (cuda/mps/cpu) for the inferences
-> gets all of the supported class names for our pretrained model
'''
class YoloV8ImageObjectDetection:
    # Path to a model. yolov8n.pt means download from PyTorch Hub
    PATH = os.environ.get("YOLO_WEIGHTS_PATHS", "yolo8.py")
    # confidence threshold
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESH", "0.7"))
    def __init__(self, chunked: bytes = None):
        """
        Initializes a yolov8 detector with a binary image
        Arguments:
            chunked (bytes): A binary image representation
        """
        self._bytes = chunked
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names
    def _get_device(self):
        """
        Gets best device for your system
        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    def _load_model(self):
        """
        Loads Yolo8 model from pytorch hub or a path on disk
        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(YoloV8ImageObjectDetection.PATH)
        return model
    

    '''
    By overriding __call__, we can run our class with the () syntax as discussed above. The first thing this function does is convert the raw binary image stream into a numpy array for ease of use. Then, we score the image, meaning that we will look through the image and see what we can detect. If we detect anything, we will assign labels and confidences to them! Now that we have a representation of what's in our image, we can plot the bounding boxes on the image and then return the annotated image and labels to the caller.
    '''
    async def __call__(self):
        """
        This function is called when class is executed.
        It analyzes a single image passed to its constructor
        and returns the annotated image and its labels
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            labels (list(str)): The corresponding labels that were found
        """
        frame = self._get_image_from_chunked()
        results = self.score_frame(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, set(labels)
    

    '''
    _get_image_from_chunked is a function that just takes our raw binary image, converts it to a Numpy NDArray, and then returns that to the caller.

    score_frame is a function that takes an Numpy NDArray as input and it passes it to our YOLO model. The YOLO model will go and run all of the detections on that frame. It will then return all of the results that it found to the caller.

    class_to_label is a function that takes an index as input. It then searches the class labels in our model, and returns the corresponding item for the given index to the user. Essentially, this converts an integer label to a human readable string label.

    plot_boxes takes the results from the score_frame function and the initial frame. It then loops through each result from score_frame and draws a bounding box around the coordinates in the image. It then labels that bounding box with the human-readable name from class_to_label.
    '''
    def _get_image_from_chunked(self):
        """
        Loads an openCV image from the raw image bytes passed by
        the API.
        Returns: 
            img (numpy.ndarray): opencv2 image object from the raw binary
        """
        arr = np.asarray(bytearray(self._bytes), dtype = np.uint8)
        img = cv2.imdecode(arr, -1)
        return img
    def score_frame(self, frame):
        """
        Scores a single image with a YoloV8 model
        Arguments:
            frame (numpy.ndarray): input frame in numpy/list/tuple format.
        Returns:
            results list(ultralytics.engine.results.Results)): Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(
            frame,
            conf = YoloV8ImageObjectDetection.CONF_THRESH,
            save_conf = True
        )
        return results
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        Arguments:
            x (int): numeric label
        Returns:   
            class (str): corresponding string label
        """
        return self.classes[int(x)]
    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, 
        and plots the bounding boxes and label on to the frame.
        Arguments:
            results (list(ultralytics.engine.results.Results)): contains labels and coordinates predicted by model on the given frame.
            frame (numpy.ndarray): Frame which has been scored.
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            labels (list(str)): The corresponding labels that were found
        """
        for r in results:
            boxes = r.boxes
            labels = []
            for box in boxes:
                c = box.cls 
                l = self.model.names[int(c)]
                labels.append(l)
        frame = results[0].plot()
        return frame, labels
    
