import argparse
from dotenv import load_dotenv
from ultralytics import YOLO
from dataset import download_training_dataset, download_validation_dataset
from models import validate

if __name__ == '__main__':

    load_dotenv()
    
    # create parser
    parser = argparse.ArgumentParser(description='Parsing arguments provided. It is needed to choose correct model weights')
    
    # add arguments
    parser.add_argument('--model', type=str, choices=['yolov11', ], help='Choose model version', required=True)
    
    parser.add_argument('--version', type=str, choices=['yolov11'], help='Choose model version', required=True)
    
    parser.add_argument('--mode', type=str, choices=['train','valid'], default='valid', help='Choose which mode of the model should be executed. Choosed related dataset', required=True)
    
    args = parser.parse_args()
    
    # download data if not exists
    training_data_dir = download_training_dataset(download_version=args.version)
    
    # download validation dataset
    validation_data_dir = download_validation_dataset(download_version=args.version)
    
    if args.model.startswith('yolo'):
        
        data_dir = f"{training_data_dir}/data.yaml" if args.mode == 'train' else f"{validation_data_dir}/data.yaml"
        
        validate(model_path=args.model, data_dir=f"{data_dir}/data.yaml")