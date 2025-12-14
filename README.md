# PatchCore Industrial Anomaly Detection (MVTec Transistor)

## Description

This project is an industrial anomaly detection application based on PatchCore using the Anomalib framework.  
It is designed to detect defective (NOK) vs normal (OK) industrial components in real time using unsupervised anomaly detection.

## The application provides:

- A desktop GUI built with CustomTkinter  
- Real-time folder monitoring for incoming images  
- PatchCore inference with anomaly heatmap overlays  
- Live counters for Total / OK / NOK components  
- Support for ONNX / OpenVINO deployment  
- Training and inference scripts using the MVTec AD Transistor dataset  

## The project includes:

- **Training & Export:** Train PatchCore on MVTec AD (Transistor) and export to ONNX  
- **Inference Script:** Batch inference with anomaly maps and result saving  
- **GUI Application:** Industrial-style inspection interface with live monitoring  


## Dataset

This project uses the MVTec Anomaly Detection (MVTec AD) dataset:

- **Class:** transistor  
- **Task:** Classification-based anomaly detection  

### Structure

```text
transistor/
├── train/
│   └── good/
└── test/
    ├── good/
    └── bad/


## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Demo](#demo)
4. [File Structure](#file-structure)
5. [Requirements](#requirements)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hassanfaham/PatchCore-Industrial-Anomaly-Detection.git
    cd PatchCore-Industrial-Anomaly-Detection
    ```

2. Install dependencies:
    ```bash
    pip install -r reqs
    ```

3. Make sure you have Python 3.11 installed.


## Usage

1. **Running the App**:
    - Start the application UI:
        ```bash
        python app_ui.py
        ```
    - Upload a model through the GUI (or add its path to the config file) and monitoring a folder will start for new images.
    - The app will automatically process images and display them in the canvas, showing the detection result as "OK" or "NOK".

2. **Inference**:
    - Run the `inference_test.py` script to perform inference on a folder of images, it also saves Classified images (OK / NOK labels), Anomaly map overlays and Heatmaps:
      ```bash
      python scripts/inference_test.py
      ```

    - The script supports running inference on a single model or a folder of models and images. It will save annotated results for each processed image.

3. **Training and Evaluation**:
    - To train a model, use the `training_test.py` script (with changing the paths of the model used and the folder of dataset used):
      ```bash
      python scripts/training_test.py
      ```

## Demo
**Video is found in the Assets folder**
![Demo](https://raw.githubusercontent.com/hassanfaham/PatchCore-Industrial-Anomaly-Detection/main/Assets/demo_thumbnail.jpg)



## File Structure
PatchCore-Industrial-Anomaly-Detection/
│
├── App files/
│   ├── app_ui.py                # Main GUI application
│   ├── model_manager.py         # Model upload & selection
│   ├── processing_manager.py    # Folder monitoring & inference
│   ├── thread_manager.py        # Thread handling
│   ├── fatal_error_handler.py   # Global error logging
│   ├── logger_config.py         # Loguru logger setup
│   ├── config.json              # Application configuration
│   └── purple.json              # CustomTkinter theme
│
├── Scripts/
│   ├── training_test.py         # PatchCore training & export
│   └── inference_test.py        # Batch inference script
│
├── Assets/
│   ├── demo.mp4
│   └── demo_thumbnail.jpeg
│
├── models/                      # Trained ONNX models & metadata
├── destination_images/          # Folder monitored by the app
├── reqs.txt                     # Python dependencies
└── README.md



## Requirements

- Python 3.11
- Required libraries:
    - `anomalib`
    - `torch`
    - `torchvision`
    - `opencv-python`
    - `numpy`
    - `Pillow`
    - `loguru`
    - `watchdog`
    - `customtkinter`
    - `openvino`
    - `onnx`

You can install the dependencies by running:
```bash
pip install -r reqs