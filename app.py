# app.py
import streamlit as st
from detector import YOLODetector
import cv2
import tempfile
import os
from datetime import datetime
import time

def main():
    # Page config
    st.set_page_config(
        page_title="Traffic Violation Detection System",
        page_icon="🚦",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .violation-box {
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            background-color: #ff4b4b20;
            border: 1px solid #ff4b4b;
        }
        .success-box {
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            background-color: #0bb84220;
            border: 1px solid #0bb842;
        }
        .stAlert {
            margin-top: 20px;
        }
        .violation-count {
            font-size: 24px;
            font-weight: bold;
            color: #ff4b4b;
        }
        .plate-text {
            font-family: monospace;
            font-size: 18px;
            background-color: #f0f2f6;
            padding: 5px 10px;
            border-radius: 3px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🚦 Traffic Violation Detection System")
    st.markdown("""
    This system detects traffic violations including:
    - 🏍️ Riders without helmets
    - 📝 Automatic license plate recognition
    - 🎯 Real-time tracking and logging
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        model_file = st.file_uploader("Upload YOLO model (.pt)", type=['pt'])
        input_file = st.file_uploader("Upload image/video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
            st.info("Higher values = more confident predictions")
            
            if st.button("Clear Violation History"):
                st.session_state.violation_log = []
                st.success("Violation history cleared!")
    
    # Initialize session state
    if 'violation_log' not in st.session_state:
        st.session_state.violation_log = []
    
    if model_file and input_file:
        try:
            # Save the uploaded model file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(model_file.read())
                model_path = tmp_model.name
            
            # Initialize detector
            detector = YOLODetector(model_path, conf_threshold)
            
            # Process based on file type
            file_extension = input_file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                # Process image
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_img:
                    tmp_img.write(input_file.read())
                    img_path = tmp_img.name
                
                # Display original image
                with col1:
                    st.subheader("Original Image")
                    st.image(img_path)
                
                # Process image with progress indicator
                with st.spinner("Processing image..."):
                    processed_image, results, violations, license_plates = detector.process_image(img_path)
                
                # Display results
                with col2:
                    st.subheader("Detection Results")
                    st.image(processed_image, channels="BGR")
                
                # Display violations and plate info
                if violations or license_plates:
                    st.markdown("<div class='violation-box'>", unsafe_allow_html=True)
                    st.subheader("⚠️ Violation Report")
                    
                    # Display violation count
                    num_violations = len(violations)
                    st.markdown(f"<p class='violation-count'>{num_violations} violation{'s' if num_violations != 1 else ''} detected</p>", 
                              unsafe_allow_html=True)
                    
                    if violations:
                        st.markdown("#### 🚫 Violations:")
                        for violation in violations:
                            st.markdown(f"- {violation}")
                    
                    if license_plates:
                        st.markdown("#### 📝 Violating Vehicles:")
                        for i, plate in enumerate(license_plates, 1):
                            st.markdown(f"- Vehicle {i}: <span class='plate-text'>{plate}</span>", 
                                      unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Log violation
                    violation_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'violations': violations,
                        'license_plates': license_plates,
                        'type': 'Image',
                        'filename': input_file.name
                    }
                    st.session_state.violation_log.append(violation_entry)
                else:
                    st.markdown("""
                        <div class='success-box'>
                            ✅ No violations detected in this image
                        </div>
                    """, unsafe_allow_html=True)
                
                # Cleanup
                os.unlink(img_path)
                
            elif file_extension in ['mp4', 'avi']:
                # Process video
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_video:
                    tmp_video.write(input_file.read())
                    video_path = tmp_video.name
                
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, status):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                # Process video
                try:
                    start_time = time.time()
                    output_path, video_violations, video_plates = detector.process_video_optimized(
                        video_path, 
                        display=False,
                        progress_callback=update_progress
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display video results
                    st.subheader("Processed Video")
                    
                    # Read and display the processed video
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                    video_file.close()
                    
                    # Display violation summary
                    if video_violations or video_plates:
                        st.markdown("<div class='violation-box'>", unsafe_allow_html=True)
                        st.subheader("📊 Video Analysis Summary")
                        
                        if video_violations:
                            st.markdown("#### 🚫 Violations Detected:")
                            for violation in video_violations:
                                st.markdown(f"- {violation}")
                        
                        if video_plates:
                            st.markdown("#### 📝 Detected License Plates:")
                            for plate in video_plates:
                                st.markdown(f"- <span class='plate-text'>{plate}</span>", 
                                        unsafe_allow_html=True)
                        
                        st.info(f"Processing Time: {processing_time:.2f} seconds")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Clean up
                    os.unlink(video_path)
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
            
            # Display violation history
            if st.session_state.violation_log:
                with st.expander("📋 Violation History", expanded=False):
                    st.markdown("### Recent Violations")
                    for entry in reversed(st.session_state.violation_log):
                        st.markdown("""
                            ---
                            **Time:** {}  
                            **File:** {}  
                            **Type:** {}  
                            **Violations:** {}  
                            **License Plates:** {}  
                            {}
                        """.format(
                            entry['timestamp'],
                            entry['filename'],
                            entry['type'],
                            ', '.join(entry['violations']) if entry['violations'] else 'None',
                            ', '.join(entry['license_plates']) if entry['license_plates'] else 'None',
                            f"**Processing Time:** {entry['processing_time']}" if 'processing_time' in entry else ''
                        ))
            
            # Cleanup model file
            os.unlink(model_path)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with different files or settings")
    
    else:
        st.info("Please upload both a YOLO model file and an image/video file to begin.")

if __name__ == "__main__":
    main()