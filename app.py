import cv2
import numpy as np
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from PIL import Image
import streamlit as st

# Load the YOLOv8 model
model = YOLO('best1.pt')

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def preprocess_image(image):
    """Preprocess the image for better barcode detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def try_decode(barcode_img):
    # Attempt to decode without preprocessing
    decoded_barcodes = decode(barcode_img)
    if decoded_barcodes:
        return decoded_barcodes
    
    # Preprocess the barcode image (e.g., convert to grayscale, apply threshold)
    barcode_img_gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
    _, barcode_img_thresh = cv2.threshold(barcode_img_gray, 128, 255, cv2.THRESH_BINARY)
    
    # Attempt to decode with preprocessing
    return decode(barcode_img_thresh)

def BarcodeReader(image):
    img = np.array(image)
    results = model(img.copy())
    detectedBarcodes = []
    barcode_data_set = set()  # To store unique barcode data
    
    # Extract bounding boxes
    boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Assuming class 0 is for barcodes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2, y2])
    
    # Apply Non-Maximum Suppression
    boxes = np.array(boxes)
    nms_boxes = non_max_suppression_fast(boxes, 0.3)
    
    # Decode the barcodes from the NMS filtered bounding boxes
    for (x1, y1, x2, y2) in nms_boxes:
        # Crop the barcode region from the image
        barcode_img = img[y1:y2, x1:x2]
        
        # Attempt to decode the barcode
        decoded_barcodes = try_decode(barcode_img)
        
        for barcode in decoded_barcodes:
            barcode_data = barcode.data.decode('utf-8').strip()  # Strip whitespace
            barcode_type = barcode.type
            if barcode_data not in barcode_data_set:
                barcode_data_set.add(barcode_data)
                detectedBarcodes.append((barcode_data, barcode_type))
        
        # Draw bounding box and label on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'Barcode', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not detectedBarcodes:
        detectedBarcodes.append(("Barcode Not Detected or your barcode is blank/corrupted!", ""))
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img_pil, detectedBarcodes

def main():
    st.title("Barcode Detection and Decoding")
    
    st.markdown("""
    <style>
    .title { font-size: 36px; color: #4CAF50; }
    .subtitle { font-size: 24px; color: #555555; }
    .description { font-size: 18px; color: #777777; }
    .result { font-size: 16px; }
    .container { display: flex; }
    .image { flex: 1; }
    .output { flex: 1; margin-left: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="title">Barcode Detection and Verification</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an image containing a barcode, enter the CIN number, and verify if it matches the barcode.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    with col2:
        cin_number = st.text_input("Enter CIN Number:")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_pil, barcode_data = BarcodeReader(image)

        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(img_pil, caption='Processed Image with Detected Barcodes', use_column_width=False, width=400)
            
            with col2:
                st.markdown('<p class="subtitle">Barcode Data:</p>', unsafe_allow_html=True)
                for data, barcode_type in barcode_data:
                    st.markdown(f'<p class="result">Decoded barcode data: {data} ({barcode_type})</p>', unsafe_allow_html=True)

                if cin_number and len(cin_number) == 8:
                    verified = False
                    st.markdown(f'<p class="result"><strong>Entered CIN Number:</strong> {cin_number}</p>', unsafe_allow_html=True)

                    # Assuming barcode_data is a list of tuples (data, barcode_type)
                    barcode_data, _ = barcode_data[0]  # Access first element (data, _)
                    barcode_value = barcode_data.strip().strip("'")  # Clean barcode data

                    verified = barcode_value[:min(len(barcode_value), 8)] == cin_number  # Safe slicing

                    if verified:
                        st.markdown('<p class="result" style="color: green;"><strong>CIN Number Verified!</strong></p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="result" style="color: red;"><strong>CIN Number Not Verified!</strong></p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()