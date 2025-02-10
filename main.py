import argparse
from dotenv import load_dotenv
from ultralytics import YOLO
from dataset import download_training_dataset, download_validation_dataset
from models import validate, predict

if __name__ == '__main__':

    load_dotenv()
    
    # create parser
    parser = argparse.ArgumentParser(description='Parsing arguments provided. It is needed to choose correct model weights')
    
    # add arguments
    parser.add_argument('--model', type=str, choices=['yolov11x', 'yolov9e', 'yolov8x' ], help='Choose model version', required=True)
    
    parser.add_argument('--version', type=str, choices=['yolov11', 'yolov9', 'yolov8'], help='Choose model version', required=True)
    
    parser.add_argument('--mode', type=str, choices=['train','valid', 'predict'], default='valid', help='Choose which mode of the model should be executed. Choosed related dataset', required=True)
    
    parser.add_argument('--dataset', type=int, choices=[2,4], default=1, help='Choose which dataset to evaluate model on')
        
    args = parser.parse_args()
    
    # download data if not exists
    
    training_data_dir = None
    validation_data_dir = None
    
    if args.mode == 'train':
        training_data_dir = download_training_dataset(download_version=args.version)
    
    # download validation dataset
    if args.mode == 'valid':
        validation_data_dir = download_validation_dataset(download_version=args.version, valid_dataset=args.dataset)
    
    if args.model.startswith('yolo'):
        
        data_dir = f"{training_data_dir}" if args.mode == 'train' else f"{validation_data_dir}"
        
        if args.mode == 'valid':
        
            validate(model_path=args.model, data_dir=f"{data_dir}/data.yaml", dataset=args.dataset)
            
        if args.mode == 'predict':
            
            img_dir = 'football-test-dataset-2'
            predict(model_path=args.model, img_dir=img_dir)