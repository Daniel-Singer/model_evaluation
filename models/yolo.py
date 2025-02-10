import os
from pathlib import Path
from ultralytics import YOLO

py_env = os.getenv('PY_ENV')

def validate(model_path, data_dir=None, dataset=None):
    
    
    dev_path = '../trained_models/'
    
    colab_path = '/content/drive/MyDrive/trained_models/'
    
    root_weight_path = dev_path if py_env == 'development' else colab_path
    
    dir_name = 'football_player_detection' if dataset == 2 else 'football_player_detection_3zvbc'
    
    model_path = f"{root_weight_path}{model_path}_{dir_name}/weights/best.pt"
    model = YOLO(model_path)
    
    metrics = model.val(data=data_dir)
    
    metrics.box.map

def predict(model_path, img_dir=None):
            
    print(os.path.isdir(img_dir))
    
    # create a list of image names
    
    imgs = Path(img_dir).joinpath('valid/images')
    
    print(imgs)
    
    img_names = list(imgs.glob('*.jpg'))
    
    print(img_names)
    
    dev_path = '../trained_models/'
    
    colab_path = '/content/drive/MyDrive/trained_models/'
    
    root_weight_path = dev_path if py_env == 'development' else colab_path
    
    model_path = f"{root_weight_path}{model_path}_football_player_detection/weights/best.pt"
    
    
    model = YOLO(model_path)
        
    results = model.predict(f"{img_dir}/valid/images", save=True, show_conf=False, show_labels=False, save_txt=True)
    
