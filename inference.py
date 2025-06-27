import torch
from PIL import Image
import os
from datasets import get_transform
from datasets import idx_to_class_covid, idx_to_class_mnist
import argparse
from utils import get_model
from models.lenet import LeNet



dataset_supported = ['covid', 'mnist']

def Predict_Singel_Image(model,img_path,dataset_name,process_rgb:bool = True,device='cpu'):
    model.eval()

    transform = get_transform(dataset_name,split='test')
    if process_rgb:
        img = Image.open(img_path).convert("RGB") 
    else:
        img = Image.open(img_path).convert("L") 
    
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        predicted_class = output.argmax(dim=1).item()

    return predicted_class

def Predict_batch_Image(model,data_path,dataset_name,process_rgb:bool = True,device='cpu'):
    model.eval()
    transform = get_transform(dataset_name,split='test')
    results = {}
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path,img_name)
        if not os.path.isfile(img_path):
            continue
        
        if process_rgb:
            img = Image.open(img_path).convert("RGB") 
        else:
            img = Image.open(img_path).convert("L") 

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
        results[img_name] = predicted_class
    return results
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit', help='Model name (default: vit)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint (.pth) file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Image not found: {args.test_data_path}")

    model = get_model(model_name=args.model_name, num_classes= args.num_classes, pre_trained=True, pretrained_path=args.checkpoint_path)
    
    model.to(device)

    dataset = args.dataset.lower()
    assert dataset in dataset_supported, f"Dataset {dataset} is not supported. Supported datasets: {dataset_supported}"
    if dataset == 'covid':
        if os.path.isfile(args.test_data_path):
            result = idx_to_class_covid[Predict_Singel_Image(model, args.test_data_path,dataset,process_rgb=True, device=device)]
            print(f"Prediction Result: Class {result} ")
        else:
            result = Predict_batch_Image(model, args.test_data_path, dataset, process_rgb=True, device=device)
            for img_name, pred in result.items():
                result[img_name] = idx_to_class_covid[pred]
            print("Batch Prediction Results:")
            for img_name, pred in result.items():
                print(f"{img_name}: Class {pred}")
            exit(0)
        
    elif dataset == 'mnist':
        if os.path.isfile(args.test_data_path):
            result = idx_to_class_mnist[Predict_Singel_Image(model, args.test_data_path, dataset,process_rgb=False, device=device)]
            print(f"Prediction Result: number {result} ")
        else:
            result = Predict_batch_Image(model, args.test_data_path, dataset, process_rgb=False, device=device)
            for img_name, pred in result.items():
                result[img_name] = idx_to_class_mnist[pred]
            print("Batch Prediction Results:")
            for img_name, pred in result.items():
                print(f"{img_name}: number {pred}")
            exit(0)
    else:
        raise ValueError(f"Dataset {dataset} is not supported yet")
    

    
    
