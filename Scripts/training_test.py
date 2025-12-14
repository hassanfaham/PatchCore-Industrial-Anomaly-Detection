import os
from anomalib import TaskType
from anomalib.models import Patchcore
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.callbacks import GraphLogger
from pytorch_lightning import Trainer


os.environ["ANOMALIB_PATCHCORE_DEVICE"] = "cpu"
import torch
torch.cuda.empty_cache()


dataset_root = r"pathto\transistor_dataset"
image_metrics = [
    "AUROC",
    "AUPR",
    "F1Score",
    "F1Max",
    "F1AdaptiveThreshold",
]


#======== instantiate/upload the data
datamodule = Folder(
    name="patchcore_training_run",
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test/bad",
    normal_test_dir="test/good",
    test_split_mode=TestSplitMode.FROM_DIR,
    # test_split_ratio = 0.2,
    val_split_mode=ValSplitMode.SAME_AS_TEST,
    val_split_ratio = 0.6,
    task=TaskType.CLASSIFICATION,
    train_batch_size=2,
    # eval_batch_size=16,
    image_size = (224, 224),
    num_workers=0,
    # normal_split_ratio = 0.2
    seed=42,
)


model = Patchcore(
    backbone='resnet18',
    layers=['layer2','layer3'],
    pre_trained=True,
    coreset_sampling_ratio=0.01,

)

#=========== instantiate the callback for chepoint and early stopping
callbacks = [
    GraphLogger(),
]

#=========== instantiate the logger
logger = AnomalibTensorBoardLogger(save_dir="TB_anomaly_detection_PATCHCORE", name="tb_experiment_1")


#=========== instantiate the engine
engine = Engine(
    logger=logger,
    accelerator="auto",
    # accelerator="gpu", devices=1,
    callbacks=callbacks,
    max_epochs=5,
    task=TaskType.CLASSIFICATION,
    image_metrics=image_metrics,
    threshold="F1AdaptiveThreshold",
    log_every_n_steps=1
)


if __name__ == "__main__":

    import os

    datamodule.setup()

    print("Fit...")
    engine.fit(datamodule=datamodule, model=model)

    print("Test...")
    test_results = engine.test(datamodule=datamodule, model=model)

    with open("metrics.txt", "w") as f:
        for metric_name, metric_value in test_results[0].items():
            f.write(f"{metric_name}: {metric_value}\n")

    print("Export weights...")
    engine.export(
        export_type=ExportType.ONNX, 
        model=model, 
        export_root="weights"
        )
    
    print(engine.model.image_threshold.cpu())




