from ultralytics import YOLO

root_weight_path = 'content/drive/MyDrive/trained_models/'


def validate(model_path, data_dir=None):
    
    # TODO needs to be fixed (the x in model path)
    model_path = f"{root_weight_path}{model_path}x_football_player_detection/weights/best.pt"
    model = YOLO(model_path)
    
    metrics = model.val(data=data_dir)
    
    metrics.box.map

    