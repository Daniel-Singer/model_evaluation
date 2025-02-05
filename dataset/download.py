import os
import sys
from roboflow import Roboflow


def download_training_dataset(download_version):
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    
    dataset_location = 'content/model_evaluation/Football-Player-Detection-8'
    dataset = None
    # check if dataset exists alread
    if not os.path.exists(dataset_location):
        rf = Roboflow(api_key)
        project = rf.workspace("augmented-startups").project("football-player-detection-kucab")
        version = project.version(8)
        dataset = version.download(download_version)
        return dataset.location
    
    return dataset_location

def download_validation_dataset(download_version):
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    
    dataset_location = 'content/model_evaluation/Football-Player-Detection-8'
    dataset = None
    # check if dataset exists alread
    if not os.path.exists(dataset_location):
        rf = Roboflow(api_key)
        project = rf.workspace("vigorelli").project("football-test-dataset")
        version = project.version(1)
        dataset = version.download(download_version)
        return dataset.location
    
    return dataset_location

