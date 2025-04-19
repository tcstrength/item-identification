import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.neighbors import NearestNeighbors
import torch

# Page config
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Interactive Object Detection")
st.markdown("Upload an image and detect objects using multiple approaches")

# Sidebar for approach selection
st.sidebar.header("Detection Settings")
detection_approach = st.sidebar.radio(
    "Select Detection Approach:",
    ["kNN with Embedded Product Images",
     "Pure Object Detection",
     "Region Proposal + Classification"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Create placeholders for the demo functions
# In a real application, these would call actual models or APIs

def generate_random_colors(n):
    """Generate n random colors for bounding box visualization"""
    colors = []
    for i in range(n):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    return colors

def draw_boxes(image, boxes, labels, scores, colors):
    """Draw bounding boxes on the image"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=f'rgb{colors[i % len(colors)]}',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x, y-10,
            f"{label}: {score:.2f}",
            color=f'rgb{colors[i % len(colors)]}',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7)
        )

    ax.axis('off')
    return fig

# Mock implementations of detection methods
def knn_detection(image, confidence):
    """Mock implementation of kNN-based detection"""
    # In a real app, this would:
    # 1. Extract features from uploaded image
    # 2. Find nearest neighbors in a pre-computed embedding space
    # 3. Return matches with bounding boxes

    st.info("Processing with kNN-based detection...")
    time.sleep(1)  # Simulate processing time

    height, width = image.shape[:2]

    # Demo boxes with product-like labels
    boxes = [
        [width*0.1, height*0.2, width*0.3, height*0.4],
        [width*0.5, height*0.3, width*0.2, height*0.3],
        [width*0.6, height*0.6, width*0.25, height*0.25]
    ]

    labels = ["Nike Shoe", "Adidas Shirt", "Ray-Ban Sunglasses"]
    scores = [0.95, 0.87, 0.76]

    # Filter by confidence
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    return filtered_boxes, filtered_labels, filtered_scores

def object_detection(image, confidence):
    """Mock implementation of generic object detection"""
    # In a real app, this would use a model like YOLO, SSD, or Faster R-CNN

    st.info("Processing with pure object detection...")
    time.sleep(1)  # Simulate processing time

    height, width = image.shape[:2]

    # Demo boxes with general object labels
    boxes = [
        [width*0.2, height*0.1, width*0.3, height*0.3],
        [width*0.1, height*0.5, width*0.4, height*0.4],
        [width*0.6, height*0.2, width*0.3, height*0.5],
        [width*0.7, height*0.7, width*0.2, height*0.2]
    ]

    labels = ["Person", "Car", "Dog", "Chair"]
    scores = [0.98, 0.85, 0.92, 0.73]

    # Filter by confidence
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    return filtered_boxes, filtered_labels, filtered_scores

def region_proposal_classification(image, confidence):
    """Mock implementation of region proposal + classification"""
    # In a real app, this would:
    # 1. Generate region proposals (e.g., using Selective Search or RPN)
    # 2. Classify each region using a separate classifier

    st.info("Processing with region proposal + classification...")
    time.sleep(1.5)  # Simulate longer processing time

    height, width = image.shape[:2]

    # Demo boxes with more detailed labels
    boxes = [
        [width*0.1, height*0.1, width*0.2, height*0.3],
        [width*0.4, height*0.2, width*0.3, height*0.4],
        [width*0.2, height*0.6, width*0.25, height*0.3],
        [width*0.6, height*0.5, width*0.3, height*0.4],
        [width*0.7, height*0.2, width*0.2, height*0.2]
    ]

    labels = ["Sports Car", "Siamese Cat", "Wooden Table", "Leather Sofa", "Smartphone"]
    scores = [0.89, 0.92, 0.78, 0.85, 0.95]

    # Filter by confidence
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    return filtered_boxes, filtered_labels, filtered_scores

# Main UI for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Display original image
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Process the image based on selected approach
    if st.button("Detect Objects"):
        with st.spinner("Processing..."):
            if detection_approach == "kNN with Embedded Product Images":
                boxes, labels, scores = knn_detection(image_np, confidence_threshold)
                approach_description = """
                **kNN Approach**: This method uses features extracted from the image to find similar
                products in a database using k-Nearest Neighbors. It's great for product recognition
                and retrieval scenarios.
                """

            elif detection_approach == "Pure Object Detection":
                boxes, labels, scores = object_detection(image_np, confidence_threshold)
                approach_description = """
                **Object Detection Approach**: A direct object detection model (like YOLO, SSD, or Faster R-CNN)
                is used to identify objects and their locations in a single pass.
                """

            else:  # Region Proposal + Classification
                boxes, labels, scores = region_proposal_classification(image_np, confidence_threshold)
                approach_description = """
                **Region Proposal + Classification**: This two-stage approach first identifies regions
                that might contain objects, then classifies each region separately. This can provide
                more detailed classification but may be slower.
                """

            st.markdown(approach_description)

            if len(boxes) > 0:
                # Generate colors for visualization
                colors = generate_random_colors(len(boxes))

                # Display results
                st.subheader("Detection Results")
                fig = draw_boxes(image_np, boxes, labels, scores, colors)
                st.pyplot(fig)

                # Show detection details in a table
                st.subheader("Detected Objects")
                result_data = {
                    "Label": labels,
                    "Confidence": [f"{score:.2f}" for score in scores],
                    "Position": [f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})" for box in boxes]
                }
                st.dataframe(result_data)
            else:
                st.warning("No objects detected with the current confidence threshold.")

    # Add explanation of the approaches
    with st.expander("About the Detection Approaches"):
        st.markdown("""
        ### Detection Approaches

        #### kNN with Embedded Product Images
        This approach works by:
        1. Extracting visual features from the uploaded image
        2. Finding the most similar images in a pre-embedded database using k-Nearest Neighbors
        3. Transferring bounding box annotations from matched images

        Best for: Product recognition, visual search, and retrieval tasks

        #### Pure Object Detection
        This approach uses end-to-end object detection models like:
        - YOLO (You Only Look Once)
        - SSD (Single Shot Detector)
        - Faster R-CNN

        These models directly predict object classes and bounding boxes in a single pass.

        Best for: General object detection with predefined categories

        #### Region Proposal + Classification
        This two-stage approach:
        1. First generates region proposals (areas likely to contain objects)
        2. Then classifies each region independently

        Benefits:
        - Can work with dynamic or custom categories
        - May provide more precise classifications
        - Adaptable to new object types

        Best for: Scenarios requiring fine-grained classification or custom categories
        """)

else:
    # Show sample images
    st.info("Upload an image to get started, or use one of our sample images below.")

    # In a real app, you would have actual sample images
    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        if st.button("Sample: Street Scene"):
            # This would load a sample image in a real app
            st.warning("Sample functionality not implemented in this demo.")

    with sample_col2:
        if st.button("Sample: Retail Products"):
            st.warning("Sample functionality not implemented in this demo.")

    with sample_col3:
        if st.button("Sample: Indoor Scene"):
            st.warning("Sample functionality not implemented in this demo.")
