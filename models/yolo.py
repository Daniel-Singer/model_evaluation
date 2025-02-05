from ultralytics import YOLO

root_weight_path = '../weights/'


def validate(model_path, data_dir=None):
    
    model_path = f"{root_weight_path}{model_path}/best.pt"
    model = YOLO(model_path)
    
    metrics = model.val(data=data_dir)
    
    metrics.box.map

    