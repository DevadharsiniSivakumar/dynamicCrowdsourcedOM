import streamlit as st
from ultralytics import YOLO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import os
import folium
from streamlit_folium import st_folium
import pandas as pd

# --- Streamlit Cloud: Load Firebase credentials from secrets TOML ---
firebase_config = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    "universe_domain": st.secrets["firebase"]["universe_domain"],
}
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
db = firestore.client()

st.title("Dynamic Crowdsourced Outdoor Obstacle Mapping")

# --- Sidebar navigation ---
page = st.sidebar.radio("Select a Page", ["📤 Upload & Detect", "🗺️ Obstacle Map"])

# --- Page 1: Upload & Detect ---
if page == "📤 Upload & Detect":
    st.header("Upload Outdoor Image for Obstacle Detection")
    st.write(
        "Upload an outdoor photo (road, street, park), and enter your location as latitude,longitude (e.g. 12.9716,77.5946). "
        "You can get these from Google Maps, or if running online, allow browser GPS for auto-fill."
    )

    # Location input (manual while local, auto possible on Streamlit Cloud)
    location = st.text_input(
        "Enter your location (GPS 'latitude,longitude', ex: 12.9716,77.5946):"
    )

    uploaded_file = st.file_uploader("Upload Your Outdoor Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        model = YOLO('yolov8n.pt')
        results = model(image)
        st.write("Detected obstacles in your photo:")

        detected_labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            st.write(f"- {label}")
            detected_labels.append(label)

        st.image(results[0].plot(), caption="Obstacles Detected", use_column_width=True)

        # Save to Firebase
        if st.button("Save Detection to Map"):
            if location.strip() == "":
                st.warning("Please enter a location before saving.")
            else:
                data = {
                    "filename": uploaded_file.name,
                    "location": location,
                    "detected_obstacles": detected_labels
                }
                db.collection("detections").add(data)
                st.success("Detection and location saved to Firebase for mapping!")

    # Optional: Show saved data table
    if st.checkbox("Show crowdsourced obstacle database"):
        docs = db.collection("detections").stream()
        rows = []
        for doc in docs:
            row = doc.to_dict()
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
        else:
            st.info("No detections have been saved yet.")

# --- Page 2: Obstacle Map ---
elif page == "🗺️ Obstacle Map":
    st.header("Live Crowdsourced Obstacle Map")
    st.write("View all detected outdoor obstacles contributed by the crowd on an interactive map.")
    docs = db.collection("detections").stream()
    rows = []
    for doc in docs:
        row = doc.to_dict()
        try:
            latlon = [float(x.strip()) for x in row["location"].split(",")]
            if len(latlon) == 2:
                row["lat"], row["lon"] = latlon
                rows.append(row)
        except Exception:
            continue

    if rows:
        map_df = pd.DataFrame(rows)
        start_coords = [map_df.iloc[0]["lat"], map_df.iloc[0]["lon"]]
        m = folium.Map(location=start_coords, zoom_start=12)
        for _, row in map_df.iterrows():
            obsts = ", ".join(row["detected_obstacles"]) if isinstance(row["detected_obstacles"], list) else row["detected_obstacles"]
            folium.Marker(
                [row["lat"], row["lon"]],
                popup=f"Obstacles: {obsts}<br>File: {row['filename']}"
            ).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.info("No locations with valid latitude/longitude found yet. Please enter coordinates in the form 'latitude,longitude'.")

