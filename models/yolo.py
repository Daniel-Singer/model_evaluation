import os
from pathlib import Path
from ultralytics import YOLO

py_env = os.getenv('PY_ENV')

def validate(model_path, data_dir=None):
    
    
    dev_path = '../trained_models/'
    
    colab_path = '/content/drive/MyDrive/trained_models/'
    
    root_weight_path = dev_path if py_env == 'development' else colab_path
    
    model_path = f"{root_weight_path}{model_path}_football_player_detection/weights/best.pt"
    model = YOLO(model_path)
    
    metrics = model.val(data=data_dir)
    
    metrics.box.map

def predict(model_path, img_dir=None):
            
    print(os.path.isdir(img_dir))
    
    # create a list of image names
    
    imgs = Path(img_dir)
    
    img_names = list(imgs.glob('*.jpg'))
    
    dev_path = '../trained_models/'
    
    colab_path = '/content/drive/MyDrive/trained_models/'
    
    root_weight_path = dev_path if py_env == 'development' else colab_path
    
    model_path = f"{root_weight_path}{model_path}_football_player_detection/weights/best.pt"
    
    model = YOLO(model_path)
    
    result = model(['football-test-dataset-2/valid/images/img_0798_jpg.rf.8e7a35f3edf728b4987c0b8898fcb264.jpg'])
    
    print(result)