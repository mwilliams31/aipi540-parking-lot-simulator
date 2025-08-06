import os
import json
import base64
import random
import concurrent.futures

import cv2
import torch

import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image
from io import BytesIO
from sklearn.metrics import f1_score


# Paths
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
DL_MODEL_PATH = os.path.join(WEBAPP_DIR, 'models/efficientnet-v2-s.pth')
SVM_MODEL_PATH = os.path.join(WEBAPP_DIR, 'models/svm-model.joblib')
SVM_SCALER_PATH = os.path.join(WEBAPP_DIR, 'models/svm-scaler.joblib')
TEST_SET_PATH = os.path.join(WEBAPP_DIR, 'data/test-files.json')
RAW_DATA_DIR = os.path.join(WEBAPP_DIR, 'data/raw')


def get_model_size(path: str) -> str:
    """Function to get the size of a model file in MB.
    
    Args:
        path (str): Path to the model file.
    
    Returns:
        str: Size of the model file in MB, formatted to two decimal places."""
    if not os.path.exists(path):
        return 'N/A'
    
    size_bytes = os.path.getsize(path)
    
    return f'{size_bytes / (1024 * 1024):.2f} MB'


def run_model_in_subprocess(model_type: str, sim_files: list, WEBAPP_DIR: str, DL_MODEL_PATH: str, SVM_MODEL_PATH: str, SVM_SCALER_PATH: str) -> dict:
    """
    Function to run the model in a separate subprocess, which allows us to manage memory usage more effectively.
    This function loads the model, processes the camera feeds, and returns the results.

    Args:
        model_type (str): Type of model to run ('dl' for deep learning, 'svm' for SVM).
        sim_files (list): List of XML files containing camera feed data.
        WEBAPP_DIR (str): Directory where the webapp is located.
        DL_MODEL_PATH (str): Path to the deep learning model.
        SVM_MODEL_PATH (str): Path to the SVM model.
        SVM_SCALER_PATH (str): Path to the SVM scaler.

    Returns:
        dict: Dictionary containing predictions, labels, total spots, inference time, and image metadata.
    """
    print(f'Starting subprocess for {model_type.upper()} model')
    import os, psutil, time, joblib, numpy as np, torch, spot_utils
    from torchvision import models, transforms
    
    print(f'Subprocess started - Running {model_type.upper()} model on {len(sim_files)} camera feeds')

    device = torch.device('cpu')

    IMAGE_SIZE = (224, 224)
    SVM_IMAGE_SIZE = (64, 128)
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
       
    # Load models
    print(f'Loading {model_type.upper()} model')
    if model_type == 'dl':
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 2)
        )
        
        model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f'EfficientNetV2 model loaded successfully')
    else:
        svm_model = joblib.load(SVM_MODEL_PATH)
        scaler = joblib.load(SVM_SCALER_PATH)
        print(f'SVM model and scaler loaded successfully')
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss
    peak_memory = baseline_memory
    memory_stages = {}

    all_preds, all_labels = [], []
    total_spots = 0
    total_inference_time = 0
    image_metadata = []

    print(f'Starting camera feed processing')
    start_simulation_time = time.time()
    for camera_idx, xml_path in enumerate(sim_files):
        print(f'Processing camera {camera_idx+1}/{len(sim_files)}: {os.path.basename(xml_path)}')

        # Load XML data
        spots, labels, contours = spot_utils.get_spot_data_from_xml(xml_path, model_type, WEBAPP_DIR, SVM_IMAGE_SIZE)
        
        # Memory usage after loading data
        mem_data_loading = process.memory_info().rss
        memory_stages[f'{camera_idx}_data_loading'] = mem_data_loading
        peak_memory = max(peak_memory, mem_data_loading)

        if not spots:
            continue

        # Extract features and make predictions
        if model_type == 'svm':
            features = spot_utils.extract_hog_features(
                spots, 
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK
            )

            # Memory usage after feature extraction
            mem_feature_extraction = process.memory_info().rss
            memory_stages[f'{camera_idx}_feature_extraction'] = mem_feature_extraction
            peak_memory = max(peak_memory, mem_feature_extraction)

            # Scale features and make predictions
            scaled_features = scaler.transform(features)
            predictions = svm_model.predict(scaled_features)
        else:
            # Deep Learning model inference
            test_transforms = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            batch = torch.stack([test_transforms(img) for img in spots]).to(device)
            with torch.no_grad():
                outputs = model(batch)
                _, preds = torch.max(outputs, 1)
            
            predictions = preds.cpu().numpy()

        # Memory usage after inference
        mem_inference = process.memory_info().rss
        memory_stages[f'{camera_idx}_inference'] = mem_inference
        peak_memory = max(peak_memory, mem_inference)

        # Add predictions and labels to overall counters
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels)
        total_spots += len(labels)

        # Calculate incorrect predictions for this camera
        camera_incorrect_predictions = sum(p != l for p, l in zip(predictions, labels))
        
        # Extract image metadata without storing the actual image
        image_path = xml_path.replace('.xml', '.jpg')
        if not os.path.isabs(image_path):
            image_path = os.path.join(WEBAPP_DIR, image_path)

        # Extract metadata
        metadata = spot_utils.extract_image_metadata(image_path)
        camera_name = metadata['camera_name']
        weather = metadata['weather']
        date = metadata['date']
            
        # Make sure contours are serializable by converting numpy arrays to lists
        serializable_contours = []
        for contour in contours:
            if isinstance(contour, np.ndarray):
                serializable_contours.append(contour.tolist())
            else:
                serializable_contours.append(contour)
            
        # Store metadata and predictions
        image_metadata.append({
            'image_path': image_path,
            'name': os.path.basename(image_path),
            'camera_id': camera_idx+1,
            'incorrect_count': camera_incorrect_predictions,
            'total_spots': len(labels),
            'camera_name': camera_name,
            'weather': weather,
            'date': date,
            'predictions': predictions.tolist(),
            'labels': labels,
            'contours': serializable_contours
        })
    
    end_simulation_time = time.time()
    total_inference_time = (end_simulation_time - start_simulation_time) * 1000  # ms

    print(f'All camera feeds processed. Total spots: {total_spots}, Total time: {total_inference_time:.2f} ms')
    print(f'Subprocess complete - Peak memory: {(peak_memory - baseline_memory) / (1024 * 1024):.2f} MB')
    
    return {
        'all_preds': all_preds,
        'all_labels': all_labels,
        'total_spots': total_spots,
        'total_inference_time': total_inference_time,
        'image_metadata': image_metadata,
        'peak_memory_usage': (peak_memory - baseline_memory) / (1024 * 1024),  # MB
        'memory_stages': memory_stages
    }

# --- Streamlit UI ---
st.set_page_config(
    page_title='Parking Lot Occupancy Simulator', 
    page_icon='🅿️',
    layout='wide'
)
st.title('🅿️ Parking Lot Occupancy - Edge Device Simulation')
st.markdown('This app simulates the performance of machine learning models running on a CPU-only edge device. It processes multiple camera feeds to provide a real-time overview of a simulated parking lot.')

device = torch.device('cpu')

# Load test files from cache, if available
if 'test_files' not in st.session_state:       
    try:
        with open(TEST_SET_PATH, 'r') as f:
            st.session_state.test_files = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f'Error loading test files from {TEST_SET_PATH}. Please ensure the file exists and is valid JSON.')
    else:
        print(f'Loaded test files - Found {len(st.session_state.test_files)} test files')
else:
    print(f'Using {len(st.session_state.test_files)} cached test files.')

# Use the cached test files
test_files = st.session_state.test_files

# Sidebar configuration
st.sidebar.header('Camera Selection')
num_cameras = st.sidebar.slider('Number of Simulated Cameras:', 
                                min_value=5, 
                                max_value=30, 
                                value=5,
                                step=5)

# Load available models
available_models = []
if os.path.exists(DL_MODEL_PATH):
    available_models.append('dl')
if os.path.exists(SVM_MODEL_PATH) and os.path.exists(SVM_SCALER_PATH):
    available_models.append('svm')

st.sidebar.info('Both models will be run on the same subset of randomly selected test files.')

if st.sidebar.button('Run Simulation', use_container_width=True):
    print(f'Simulation started - Processing {num_cameras} camera feeds')
    
    if num_cameras > 0 and test_files:
        sim_files = random.sample(test_files, num_cameras)
    else:
        raise ValueError('No test files available or invalid number of cameras selected.')

    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future_to_model = {
            executor.submit(
                run_model_in_subprocess,
                model_type,
                sim_files,
                WEBAPP_DIR,
                DL_MODEL_PATH,
                SVM_MODEL_PATH,
                SVM_SCALER_PATH
            ): model_type for model_type in available_models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            model_type = future_to_model[future]
            try:
                res = future.result()
                results[model_type] = res
                print(f'{model_type.upper()} model finished - {res["total_spots"]} spots processed')
            except Exception as exc:
                print(f'{model_type.upper()} model generated an exception: {exc}')

    st.success(f'Processed {len(sim_files)} camera feeds.')

    # Create scorecard
    print(f'Creating scorecard')
    st.header('Model Scorecard')
    comparison_data = {}
    for model_type in available_models:
        # Get results for the current model
        all_preds = results[model_type]['all_preds']
        all_labels = results[model_type]['all_labels']
        total_spots = results[model_type]['total_spots']
        
        # Calculate accuracy
        correct_predictions = sum(p == l for p, l in zip(all_preds, all_labels))
        incorrect_predictions = total_spots - correct_predictions
        accuracy = (correct_predictions / total_spots) * 100 if total_spots > 0 else 0
        
        # Calculate F1 score for the occupied class
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0) * 100  # Convert to percentage
        
        # Get processing time and memory usage
        processing_time = results[model_type]['total_inference_time']
        time_per_spot = processing_time / total_spots if total_spots > 0 else 0
        model_path = DL_MODEL_PATH if model_type == 'dl' else SVM_MODEL_PATH
        model_size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        model_size_mb = model_size_bytes / (1024 * 1024)
        memory_usage = results[model_type]['peak_memory_usage']
        
        comparison_data[model_type] = {
            'incorrect_predictions': incorrect_predictions,
            'total_spots': total_spots,
            'accuracy': accuracy,
            'f1_score': f1,
            'processing_time': processing_time,
            'model_size': model_size_mb,
            'memory_usage': memory_usage,
            'time_per_spot': time_per_spot
        }

    # Add a winner column to show which model performs better
    winner_column = []

    # Determine Incorrect Predictions winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_incorrect = comparison_data['dl']['incorrect_predictions']
        svm_incorrect = comparison_data['svm']['incorrect_predictions']
        
        if dl_incorrect < svm_incorrect:
            pct_diff = ((svm_incorrect - dl_incorrect) / svm_incorrect) * 100 if svm_incorrect > 0 else 100
            winner_column.append(f'EfficientNetV2 (-{pct_diff:.1f}%)')
        elif svm_incorrect < dl_incorrect:
            pct_diff = ((dl_incorrect - svm_incorrect) / dl_incorrect) * 100 if dl_incorrect > 0 else 100
            winner_column.append(f'SVM with HOG (-{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')
        
    # Determine F1 Score winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_f1 = comparison_data['dl']['f1_score']
        svm_f1 = comparison_data['svm']['f1_score']
        if dl_f1 > svm_f1:
            pct_diff = ((dl_f1 - svm_f1) / svm_f1) * 100 if svm_f1 > 0 else dl_f1
            winner_column.append(f'EfficientNetV2 (+{pct_diff:.1f}%)')
        elif svm_f1 > dl_f1:
            pct_diff = ((svm_f1 - dl_f1) / dl_f1) * 100 if dl_f1 > 0 else svm_f1
            winner_column.append(f'SVM with HOG (+{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')
    
    # Determine Total Processing Time winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_time = comparison_data['dl']['processing_time']
        svm_time = comparison_data['svm']['processing_time']
        if dl_time < svm_time:
            pct_diff = ((svm_time - dl_time) / svm_time) * 100
            winner_column.append(f'EfficientNetV2 (-{pct_diff:.1f}%)')
        elif svm_time < dl_time:
            pct_diff = ((dl_time - svm_time) / dl_time) * 100
            winner_column.append(f'SVM with HOG (-{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')
    
    # Determine Time per Spot winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_spot_time = comparison_data['dl']['time_per_spot']
        svm_spot_time = comparison_data['svm']['time_per_spot']
        if dl_spot_time < svm_spot_time:
            pct_diff = ((svm_spot_time - dl_spot_time) / svm_spot_time) * 100
            winner_column.append(f'EfficientNetV2 (-{pct_diff:.1f}%)')
        elif svm_spot_time < dl_spot_time:
            pct_diff = ((dl_spot_time - svm_spot_time) / dl_spot_time) * 100
            winner_column.append(f'SVM with HOG (-{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')

    # Determine Model Size winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_size = comparison_data['dl']['model_size']
        svm_size = comparison_data['svm']['model_size']
        if dl_size < svm_size:
            pct_diff = ((svm_size - dl_size) / svm_size) * 100
            winner_column.append(f'EfficientNetV2 (-{pct_diff:.1f}%)')
        elif svm_size < dl_size:
            pct_diff = ((dl_size - svm_size) / dl_size) * 100
            winner_column.append(f'SVM with HOG (-{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')
    
    # Determine Peak RAM Usage winner
    if 'dl' in comparison_data and 'svm' in comparison_data:
        dl_ram = comparison_data['dl']['memory_usage']
        svm_ram = comparison_data['svm']['memory_usage']
        if dl_ram < svm_ram:
            pct_diff = ((svm_ram - dl_ram) / svm_ram) * 100
            winner_column.append(f'EfficientNetV2 (-{pct_diff:.1f}%)')
        elif svm_ram < dl_ram:
            pct_diff = ((dl_ram - svm_ram) / dl_ram) * 100
            winner_column.append(f'SVM with HOG (-{pct_diff:.1f}%)')
        else:
            winner_column.append('Tie (0%)')
    else:
        winner_column.append('N/A')

    # Create a DataFrame for the scorecard
    df = pd.DataFrame({
        'Metric': [
            'Incorrect Predictions',
            'F1 Score (Occupied Class)',
            'Total Processing Time',
            'Time per Spot',
            'Model Size',
            'Peak RAM Usage'
        ],
        'EfficientNetV2': [
            f'{comparison_data["dl"]["incorrect_predictions"]} out of {comparison_data["dl"]["total_spots"]} ({comparison_data["dl"]["accuracy"]:.1f}%)' if 'dl' in comparison_data else 'N/A',
            f'{comparison_data["dl"]["f1_score"]:.3f}%' if 'dl' in comparison_data else 'N/A',
            f'{comparison_data["dl"]["processing_time"] / 1000:.2f} s' if 'dl' in comparison_data else 'N/A',
            f'{comparison_data["dl"]["time_per_spot"]:.2f} ms' if 'dl' in comparison_data else 'N/A',
            f'{comparison_data["dl"]["model_size"]:.2f} MB' if 'dl' in comparison_data else 'N/A',
            f'{comparison_data["dl"]["memory_usage"]:.2f} MB' if 'dl' in comparison_data else 'N/A'
        ],
        'SVM with HOG': [
            f'{comparison_data["svm"]["incorrect_predictions"]} out of {comparison_data["svm"]["total_spots"]} ({comparison_data["svm"]["accuracy"]:.1f}%)' if 'svm' in comparison_data else 'N/A',
            f'{comparison_data["svm"]["f1_score"]:.3f}%' if 'svm' in comparison_data else 'N/A',
            f'{comparison_data["svm"]["processing_time"] / 1000:.2f} s' if 'svm' in comparison_data else 'N/A',
            f'{comparison_data["svm"]["time_per_spot"]:.2f} ms' if 'svm' in comparison_data else 'N/A',
            f'{comparison_data["svm"]["model_size"]:.2f} MB' if 'svm' in comparison_data else 'N/A',
            f'{comparison_data["svm"]["memory_usage"]:.2f} MB' if 'svm' in comparison_data else 'N/A'
        ],
        'Winner': winner_column
    })

    # Use a function to highlight the winners
    def highlight_winners(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for i, winner in enumerate(df['Winner']):
            if 'EfficientNetV2' in winner and 'Tie' not in winner:
                styles.iloc[i, 1] = 'font-weight: bold; background-color: #d4f7d4;'
            elif 'SVM with HOG' in winner and 'Tie' not in winner:
                styles.iloc[i, 2] = 'font-weight: bold; background-color: #d4f7d4;'
            elif 'Tie' in winner:
                # For ties, highlight both cells with a different color
                styles.iloc[i, 1] = 'font-weight: bold; background-color: #f7f7d4;'
                styles.iloc[i, 2] = 'font-weight: bold; background-color: #f7f7d4;'
        
        return styles

    # Apply the styling and display the DataFrame
    styled_df = df.style.apply(highlight_winners, axis=None)
    st.dataframe(styled_df, use_container_width=True)
    print(f'Scorecard displayed')

    # Display model-specific performance dashboards
    print(f'Rendering model-specific dashboards')
    for model_type in available_models:
        model_name = 'EfficientNetV2 (Deep Learning)' if model_type == 'dl' else 'SVM with HOG Features'
        print(f'Rendering dashboard for {model_name}')
        st.header(f'{model_name} Performance Dashboard')
        model_results = results[model_type]
        
        # Model-specific dashboard metrics
        all_preds = model_results['all_preds']
        all_labels = model_results['all_labels']
        
        occupied_count = sum(1 for p in all_preds if p == 1)
        available_count = model_results['total_spots'] - occupied_count
        correct_predictions = sum(p == l for p, l in zip(all_preds, all_labels))
        incorrect_predictions = model_results['total_spots'] - correct_predictions
        overall_accuracy = (correct_predictions / model_results['total_spots']) * 100 if model_results['total_spots'] > 0 else 0
        
        # Use scikit-learn for F1 score
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Available Spots', f'{available_count} / {model_results["total_spots"]}')
        with col2:
            st.metric('Incorrect Predictions', f'{incorrect_predictions} out of {model_results["total_spots"]} ({overall_accuracy:.1f}%)')
        with col3:
            st.metric('F1 Score (Occupied)', f'{f1:.3f}%')
        with col4:
            st.metric('Total Processing Time', f'{model_results["total_inference_time"] / 1000:.2f} s')

        st.subheader('Efficiency Metrics (CPU-Only Edge Device)')
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            model_path = DL_MODEL_PATH if model_type == 'dl' else SVM_MODEL_PATH
            st.metric('Model Size', get_model_size(model_path))
        with m_col2:
            st.metric('Peak RAM Usage', f'{model_results["peak_memory_usage"]:.2f} MB')
        with m_col3:
            st.metric('Avg. Time per Spot', f'{model_results["total_inference_time"] / model_results["total_spots"]:.2f} ms')

        st.subheader(f'Camera Feeds - {model_name} (Green=Correct, Red=Incorrect)')
        max_images_to_display = 5
        
        # Get metadata from subprocess results
        image_metadata = model_results['image_metadata']
        
        # Sort images by number of incorrect predictions (descending)
        sorted_metadata = sorted(image_metadata, 
                              key=lambda x: x['incorrect_count'], 
                              reverse=True)
        
        # Apply maximum limit
        display_metadata = sorted_metadata[:max_images_to_display]
        
        # Create annotated images
        annotated_images = []
        for metadata in display_metadata:
            try:
                # Load the image
                image_path = metadata['image_path']
                original_image = cv2.imread(image_path)
                
                if original_image is not None:
                    # Create annotation
                    display_image = original_image.copy()
                    
                    # Draw contours using prediction data
                    predictions = metadata['predictions']
                    labels = metadata['labels']
                    contours = metadata['contours']
                    
                    for pred, true, contour in zip(predictions, labels, contours):
                        # Convert contour back to numpy array if needed
                        if not isinstance(contour, np.ndarray):
                            contour = np.array(contour, dtype=np.int32)
                        
                        color = (0, 255, 0) if pred == true else (0, 0, 255)
                        cv2.drawContours(display_image, [contour], 0, color, 2)
                    
                    # Convert BGR to RGB for display
                    annotated_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                    
                    # Store the annotated image with its metadata
                    annotated_images.append({
                        'image': annotated_rgb,
                        'name': metadata['name'],
                        'camera_id': metadata['camera_id'],
                        'incorrect_count': metadata['incorrect_count'],
                        'total_spots': metadata['total_spots'],
                        'camera_name': metadata['camera_name'],
                        'weather': metadata['weather'],
                        'date': metadata['date']
                    })
            except Exception as e:
                st.error(f"Error annotating image {metadata['name']}: {e}")
        
        # Display the annotated images
        if not annotated_images:
            st.warning(f'No images were processed successfully with the {model_name}. Check console for errors.')
        else:
            try:
                # Create tabs for each camera feed
                tab_names = [f'Camera {img_data["camera_id"]} ({img_data["incorrect_count"]} errors)' for img_data in annotated_images]
                tabs = st.tabs(tab_names)

                # Display each image in its respective tab
                for tab, img_data in zip(tabs, annotated_images):
                    with tab:
                        error_percent = (img_data['incorrect_count'] / img_data['total_spots']) * 100 if img_data['total_spots'] > 0 else 0
                        
                        # Camera feed title
                        st.write(f'**Camera Feed {img_data["camera_id"]}**')

                        # Create info section
                        info_html = f"""
                        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <b>Camera:</b> {img_data.get('camera_name', 'Unknown')}<br>
                            <b>Weather:</b> {img_data.get('weather', 'Unknown')}<br>
                            <b>Date:</b> {img_data.get('date', 'Unknown')}<br>
                            <b>Filename:</b> {img_data['name']}<br>
                            <b>Errors:</b> {img_data['incorrect_count']} out of {img_data['total_spots']} spots ({error_percent:.1f}%)
                        </div>
                        """
                        st.markdown(info_html, unsafe_allow_html=True)
                        
                        # Display the image
                        pil_img = Image.fromarray(img_data['image'])
                        buffer = BytesIO()
                        pil_img.save(buffer, format='PNG')
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; width:auto; height:auto;">'
                        st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f'Error in image display section: {str(e)}')
    st.divider()