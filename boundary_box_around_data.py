# Task
Detect and draw bounding boxes around ladybugs in images located in the "/content/output_synthetic_datasets" folder using pattern recognition.

## Prepare a dataset

### Subtask:
Gather a dataset of images containing ladybugs and annotate them with bounding boxes.

## Prepare a dataset

### Subtask:
Gather a dataset of images containing ladybugs and annotate them with bounding boxes.

## Choose an object detection model

### Subtask:
Select a suitable object detection model architecture (e.g., YOLO, Faster R-CNN).

## Implement detection

### Subtask:
Implement detection using the chosen object detection model (YOLO) to detect ladybugs in the synthetic images and obtain their bounding box coordinates.

**Reasoning**:
Implement object detection using the chosen object detection model (YOLO) to detect ladybugs in the synthetic images and obtain their bounding box coordinates.

**Reasoning**:
The previous command failed because the 'ultralytics' library is not installed. Install the 'ultralytics' library using pip to resolve the ModuleNotFoundError.
"""

!pip install ultralytics

"""**Reasoning**:
The 'ultralytics' library has been successfully installed. Now, proceed with the object detection by loading the YOLO model and processing the loaded synthetic images to extract bounding boxes.


"""

from ultralytics import YOLO

# Load the pre-trained YOLO model
# Using 'yolov8n.pt' as an example; a different model might be more suitable depending on the specific use case and dataset
model = YOLO('yolov8n.pt')

# Initialize a list to store detected bounding boxes for each image
detected_bboxes = []

# Iterate through the loaded synthetic images
for image in loaded_synthetic_images:
    # Perform object detection using the loaded YOLO model
    # Setting confidence threshold lower for potentially small or varied synthetic ladybugs
    # Adjust the 'conf' parameter as needed
    results = model(image, conf=0.25)  # Pass the image directly to the model

    # Extract bounding box coordinates
    # The results object contains information about detections
    # Accessing bounding boxes for the first result (assuming batch size 1)
    if results and len(results) > 0:
        # 'boxes' contains the bounding box coordinates, class labels, and confidence scores
        # .xyxy gives the coordinates in [x1, y1, x2, y2] format
        image_bboxes = results[0].boxes.xyxy.tolist()
        detected_bboxes.append(image_bboxes)
    else:
        # If no detections in an image, append an empty list
        detected_bboxes.append([])

# Print the number of images for which bounding boxes were processed
print(f"Processed detection for {len(detected_bboxes)} images.")

# You can inspect the first few detected bounding boxes to verify
# print("\nDetected Bounding Boxes for the first image:")
# if detected_bboxes and detected_bboxes[0]:
#     for bbox in detected_bboxes[0]:
#         print(bbox)
# else:
#      print("No bounding boxes detected for the first image.")

"""## Draw bounding boxes

### Subtask:
Draw the detected bounding boxes on the corresponding synthetic images using the obtained coordinates.

**Reasoning**:
Iterate through the loaded synthetic images and their corresponding detected bounding boxes, draw rectangles on the images if bounding boxes are present, and store the modified images.
"""

# Initialize a list to store images with drawn bounding boxes
images_with_bboxes = []

# Iterate through loaded images and detected bounding boxes
for i, image in enumerate(loaded_synthetic_images):
    # Create a copy of the image to draw on, to avoid modifying the original
    image_copy = image.copy()

    # Get the bounding boxes for the current image
    bboxes = detected_bboxes[i]

    # Check if any bounding boxes were detected for this image
    if bboxes:
        # Iterate through each bounding box
        for bbox in bboxes:
            # Bounding box format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Draw the rectangle on the image copy
            # Color in BGR format (e.g., green)
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

    # Append the image (with or without bounding boxes) to the new list
    images_with_bboxes.append(image_copy)

# Now 'images_with_bboxes' contains the original images with detected bounding boxes drawn on them.
# You can display or save these images as needed in subsequent steps.
print(f"Processed {len(images_with_bboxes)} images with bounding box drawing.")

"""## Visualize results

### Subtask:
Display the images with the drawn bounding boxes.

**Reasoning**:
Display the images with the drawn bounding boxes using cv2.imshow, wait for a key press for each image, and then close all windows.

**Reasoning**:
The previous command failed because `cv2.imshow` is disabled in Colab. The error message suggests using `cv2_imshow` from `google.colab.patches` as a substitute. Import the necessary function and use it to display the images.
"""

from google.colab.patches import cv2_imshow

# Iterate through the images with drawn bounding boxes
for i, image in enumerate(images_with_bboxes):
    # Use cv2_imshow() to display the image with the bounding box
    # cv2_imshow does not require window names or waitKey
    print(f"Displaying Image {i+1} with Bounding Box:")
    cv2_imshow(image)

# In Colab with cv2_imshow, there's no need for cv2.waitKey(0) or cv2.destroyAllWindows()
# as the images are displayed inline in the notebook output.

print("Displayed images with bounding boxes.")

"""## Summary:

### Data Analysis Key Findings

*   The initial attempt to gather and annotate a ladybug dataset manually failed as it cannot be automated within the current environment.
*   YOLO was selected as a suitable object detection model.
*   The `ultralytics` library was successfully installed to use the YOLO model.
*   Object detection was successfully implemented using YOLO, although the model detected a "donut" instead of a "ladybug" in the synthetic image, indicating the need for a model trained on ladybug images.
*   Bounding boxes were successfully drawn on the synthetic image using the detected coordinates.
*   The images with drawn bounding boxes were successfully displayed using `cv2_imshow()` in the Colab environment.

### Insights or Next Steps

*   To accurately detect ladybugs, a YOLO model specifically trained on a diverse dataset of ladybug images is required.
*   Further steps would involve training or fine-tuning a YOLO model on a ladybug dataset and then applying the detection and drawing process to the synthetic images.
"""
