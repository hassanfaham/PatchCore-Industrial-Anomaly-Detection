
import os
import json
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from anomalib.deploy.inferencers import OpenVINOInferencer
from anomalib import TaskType
from anomalib.utils.visualization.image import add_anomalous_label, add_normal_label

metadata_path = r"pathto\metadata.json"
model_path = r"pathto\model.onnx"
input_folder = r"pathto"  # Folder containing input images
output_folder = "output_img"
predicted_ok = 0
predicted_nok = 0
t_norm = 0.5

os.makedirs(output_folder, exist_ok=True)

with open(metadata_path, "r") as f:
    md = json.load(f)

inferencer = OpenVINOInferencer(
    path=model_path,
    metadata=metadata_path,
    device="GPU",
    task=TaskType.CLASSIFICATION,
)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


for image_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, image_name)

    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    image = Image.open(input_path).convert("RGB")

    prediction = inferencer.predict(image=image)
    pred_label = prediction.pred_label
    pred_score = prediction.pred_score

    if pred_score >= t_norm:
        predicted_nok += 1
    else:
        predicted_ok += 1

    if pred_label:
        image_classified = add_anomalous_label(np.array(image), pred_score)
    else:
        image_classified = add_normal_label(np.array(image), 1 - pred_score)

    # Save the classified image
    classified_dir = os.path.join(output_folder, "classified")
    os.makedirs(classified_dir, exist_ok=True)
    classified_path = os.path.join(classified_dir, f"{os.path.splitext(image_name)[0]}_classified.jpg")
    Image.fromarray(image_classified).save(classified_path, format="JPEG", quality=85)

    # Save the anomaly map overlayed on the image (if available)
    if hasattr(prediction, "anomaly_map") and prediction.anomaly_map is not None:
        anomaly_map = prediction.anomaly_map
        anomaly_map_resized = cv2.resize(anomaly_map, (np.array(image).shape[1], np.array(image).shape[0]))
        anomaly_map_normalized = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX)
        anomaly_map_uint8 = anomaly_map_normalized.astype(np.uint8)
        anomaly_map_colored = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlayed_image = cv2.addWeighted(image_bgr, 0.6, anomaly_map_colored, 0.4, 0)
        overlayed_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
        overlayed_image_pil = Image.fromarray(overlayed_image_rgb)

        # Save overlayed image
        overlayed_dir = os.path.join(output_folder, "anomaly_map_overlayed")
        os.makedirs(overlayed_dir, exist_ok=True)
        overlayed_image_path = os.path.join(overlayed_dir, f"{os.path.splitext(image_name)[0]}_overlayed.png")
        overlayed_image_pil.save(overlayed_image_path)

    # Save heatmap (if available)
    if hasattr(prediction, "heat_map") and prediction.heat_map is not None:
        heatmap_dir = os.path.join(output_folder, "heat_map")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_path = os.path.join(heatmap_dir, f"{os.path.splitext(image_name)[0]}_heatmap.png")
        heatmap_image = Image.fromarray((prediction.heat_map).astype(np.uint8))
        heatmap_image.save(heatmap_path)

    # Save segmentation (if available)
    if hasattr(prediction, "segmentations") and prediction.segmentations is not None:
        segmentation_dir = os.path.join(output_folder, "segmentation")
        os.makedirs(segmentation_dir, exist_ok=True)
        segmentation_path = os.path.join(segmentation_dir, f"{os.path.splitext(image_name)[0]}_segmentation.png")
        segmentation_image = Image.fromarray(prediction.segmentations.astype(np.uint8))
        segmentation_image.save(segmentation_path)

print(f"Inference complete. Results saved in: {output_folder}")
print(f"  OK: {predicted_ok} | NOK: {predicted_nok}")
print("\nProcessing complete!")

