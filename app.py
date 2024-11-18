import streamlit as st
import face_recognition
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import numpy as np

# Helper function to extract images from PDFs
def extract_images_from_pdf(pdf_file):
    images = []
    pdf_pages = convert_from_path(pdf_file)
    for page in pdf_pages:
        images.append(page.convert("RGB"))  # Ensure images are in RGB format
    return images

# Helper function for face recognition
def process_faces(images):
    face_encodings = []
    for img in images:
        # Ensure the image is in RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        face_locations = face_recognition.face_locations(img_array)
        encodings = face_recognition.face_encodings(img_array, face_locations)
        face_encodings.extend(encodings)
    return face_encodings

# Compare faces and group similar ones
def compare_faces(face_encodings):
    if not face_encodings:
        return []

    groups = []
    visited = set()

    for i, face in enumerate(face_encodings):
        if i in visited:
            continue
        group = [i]
        for j, other_face in enumerate(face_encodings):
            if i != j and j not in visited:
                similarity = face_recognition.compare_faces([face], other_face)
                if similarity[0]:
                    group.append(j)
                    visited.add(j)
        groups.append(group)
        visited.add(i)
    return groups

# Streamlit UI
st.title("Face Recognition in Documents")
uploaded_files = st.file_uploader("Upload PDFs or Images", accept_multiple_files=True, type=["pdf", "jpg", "jpeg", "png"])

if uploaded_files:
    all_images = []

    # Process uploaded files
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_images = extract_images_from_pdf(file)
            all_images.extend(pdf_images)
        else:
            img = Image.open(file)
            if img.mode != "RGB":  # Ensure images are in RGB format
                img = img.convert("RGB")
            all_images.append(img)

    st.write("Uploaded Files:")
    for img in all_images:
        st.image(img, width=150)

    st.write("Processing Faces...")
    face_encodings = process_faces(all_images)

    if face_encodings:
        face_groups = compare_faces(face_encodings)

        # Display grouped results
        for i, group in enumerate(face_groups):
            st.subheader(f"Group {i+1}: These documents belong to the same person.")
            for idx in group:
                st.image(all_images[idx], caption=f"Face {idx+1}", width=150)
    else:
        st.warning("No faces detected in the uploaded documents. Please try again with different files.")
