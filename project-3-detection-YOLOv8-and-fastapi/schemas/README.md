# Schemas
We will only have a single response schema in our API. Essentially, it will be a data-type class which returns a few things to the user:

```
The id of the uploaded image
The labels our detector found
```

We just have to inherit from the pydantic.BaseModel class. Pydantic will then go and do all of the necessary serialization when our API returns this ImageAnalysisResponse to the user.