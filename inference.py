import torch
from PIL import Image
from models import model_dict 
import os
from datasets import get_transform_covidradiography,idx_to_class_covid
import argparse
from utils import get_model

def Predict_Singel_Image(model,img_path,dataset_name,device='cpu'):
    model.eval()
    if dataset_name == 'covid':
        transform = get_transform_covidradiography(split='test')
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet")
    
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    
    return predicted_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit', help='Model name (default: vit)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint (.pth) file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    model = get_model(model_name=args.model_name, num_classes= args.num_classes, pre_trained=True, pretrained_path=args.checkpoint_path)

    model.to(device)

    result = idx_to_class_covid[Predict_Singel_Image(model, args.image_path,args.dataset, device)]
    

    print(f"Prediction Result: Class {result} ")
    
