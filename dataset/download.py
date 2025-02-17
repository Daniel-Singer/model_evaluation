import os
import sys
from roboflow import Roboflow


def download_training_dataset(download_version):
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    
    py_env = os.getenv('PY_ENV')
    
    print(py_env)
    
    dev_location = '../Football-Player-Detection-8'
    
    colob_location = 'content/model_evaluation/Football-Player-Detection-8'
    
    dataset_location = colob_location if py_env == 'production' else dev_location
    
    dataset = None
    # check if dataset exists alread
    if not os.path.exists(dataset_location):
        rf = Roboflow(api_key)
        project = rf.workspace("augmented-startups").project("football-player-detection-kucab")
        version = project.version(8)
        dataset = version.download(download_version)
        return dataset.location
    
    return dataset_location

def download_validation_dataset(download_version, valid_dataset=None):
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    
    py_env = os.getenv('PY_ENV')
    
    # create path string based on validation
        
    dev_location = f"../football-test-dataset-{valid_dataset}"
    
    colob_location = f"content/model_evaluation/football-test-dataset-{valid_dataset}"
    
    dataset_location = colob_location if py_env == 'production' else dev_location
    
    dataset = None
    # check if dataset exists alread
    if not os.path.exists(dataset_location):
        rf = Roboflow(api_key)
        project = rf.workspace("vigorelli").project("football-test-dataset")
        version = project.version(valid_dataset)
        dataset = version.download(download_version)
        return dataset.location
    
    return dataset_location

