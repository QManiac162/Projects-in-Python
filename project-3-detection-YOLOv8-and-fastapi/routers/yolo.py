# for API operations and stds
from fastapi import APIRouter, UploadFile, Response, status, HTTPException


# detector objects
from detectors import yolo8


# for encoding images
import cv2


# for response schemas
from schemas.yolo import ImageAnalysisResponse


# A new router object that we can add endpoints to
# note that the prefix is /yolo, so all endpoints from here on will be relative to /yolo
router = APIRouter(tags = ["Image upload and analysis"], prefix = "/yolo")


# A cache of annoted images. This would be some sort of persistent storage (postgres+S3
# but for simplicity we can keep this things in memory
'''
We are adding a new POST method to our api at /yolo/ (because the router is prefixed with /yolo). The route will return an HTTP 201 with the response body of our ImageAnalysisResponse schema. The route will also expect, as input, a multi-part upload of an image. When we enter this function, we will first read the image and then pass it to our YoloV8ImageObjectDetection object (which we will discuss in the next section). We then use the callable YoloV8ImageObjectDetection object to run our analysis, encode the image in png format, and save it in our in-memory array. Finally, we return an ImageAnalysisResponse object with the id and any detected labels filled out. At this point, we can successfully upload/analyze/save images in our application.
'''
images = []
# adding the 1st endpoint
@router.post("/",
             status_code = status.HTTP_201_CREATED,
             responses = {
                 201: {"description": "Successfully analysed the image"}
             },
             response_model = ImageAnalysisResponse,
             )
async def yolo_image_upload(file: UploadFile) -> ImageAnalysisResponse:
    """
    Takes a multi-part upload image and runs yolo8 on it to detect objects

    Arguments:
        file(UploadFile): The multi-part upload file

    Returns:
        response(ImageAnalysisResponse): The image ID and the labels in the PyDantic object

    Example curl:
        curl -X 'POST'\
            'http://localhost/yolo/'\
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'file=@image.jpg;type=image/jpeg'

    Example Return:
        {
            "id": 1,
            "labels": [
                "vase"
            ]
        }
    """
    contents = await file.read()
    dt = yolo8.YoloV8ImageObjectDetection(chunked = contents)
    frame, labels = await dt()
    success, encoded_image = cv2.imencode(".png", frame)
    images.append(encoded_image)
    return ImageAnalysisResponse(id = len(images), labels = labels)


# adding one more endpoint to download the images
'''
adding a new GET method to our router at /yolo/<id>. The route will return an HTTP 200 if the image ID is in our array, otherwise, it will return a 404. The body of the response will be a binary PNG image. The application code is a bit easier here, as all we have to do is index our array and return the encoded image.
'''
@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image ID Not Found."}
    },
    response_class=Response,
)
async def yolo_image_download(image_id: int) -> Response:
    """Takes an image id as a path param and returns that encoded
    image from the images array
    Arguments:
        image_id (int): The image ID to download
    
    Returns:
        response (Response): The encoded image in PNG format
    
    Examlple cURL:
        curl -X 'GET' \
            'http://localhost/yolo/1' \
            -H 'accept: image/png'
            
    Example Return: A Binary Image
    """
    try:
        return Response(content=images[image_id - 1].tobytes(), media_type="image/png")
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found")