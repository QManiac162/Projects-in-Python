# Routers
The entrypoint of APIs are typically their routes (aka endpoints). Our API is going to support two endpoints:

```
Upload an image and run yolo on it
Download an image that yolo annotated for us
```
So, we do some standard imports. We will also make a new router object which we will add our two endpoints to. Note that the router will be prefixed with /yolo. For example, curl calls would be

```
curl http://localhost/yolo
curl http://localhost/yolo/endpoint1
curl http://localhost/yolo/endpoint2
```
