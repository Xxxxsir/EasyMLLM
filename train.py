import os
import argparse
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import *
from utils import set_seed, get_model


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="vit")
    parser.add_argument("--dataset_name",type=str,default="covid")
    parser.add_argument("--classes_num", type=int, default=4, help="the number of classes in the dataset")
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--pre_trained", type=bool, default=False, help="whether to use pre-trained model parameters")

    return parser


def train_one_epoch(model, 
                    train_loader,
                    val_loader, 
                    optimizer,
                    device:str = "cpu",
                    criterion:torch.Tensor = F.cross_entropy):
    model.train()
    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0

    for batch_idx, (image, label) in enumerate(tqdm(train_loader,desc="Training")):

        image, label = image.to(device), label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item() * image.size(0)
        _,predicted = torch.max(output,dim = 1)
        train_total_correct += (predicted == label).sum().item()
        train_total_samples += image.size(0)

    train_loss = train_total_loss / train_total_samples
    train_accuracy = train_total_correct / train_total_samples

    model.eval()
    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(val_loader,desc="Validation")):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)

            val_total_loss += loss.item() * image.size(0)
            _, predicted = torch.max(output, dim=1)
            val_total_correct += (predicted == label).sum().item()
            val_total_samples += image.size(0)
    
    val_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return train_loss, train_accuracy, val_loss, val_accuracy


def train(model_name:str,
          dataset_name:str,
          model,
          train_loader,
          val_loader,
          epochs=10,
          lr=1e-3,
          weight_decay=1e-4,
          device="cpu",
          pre_trained=False,
          momentum=0.9,
          save_dir="./model_ckpt",
          ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )

    criterion = F.cross_entropy
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = -1.0

    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc = train_one_epoch(
            model, train_loader, val_loader, optimizer, device=device,criterion=criterion
        )
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "accuracy": best_acc
            }
            torch.save(checkpoint, os.path.join(save_dir, f"{dataset_name}_{model_name}.pth"))
            print(f"New best model saved with accuracy: {best_acc:.4f}")
    
    print(f"Training completed !")
        
    
def main():
    parser = arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = get_model(args.model_name, args.classes_num, args.pre_trained,\
                      img_size=28,patch_size=4,in_channel=1,embed_dim=128,depth=6,num_heads=4)
    model.to(device)

    print(f"Loading dataset: {args.dataset_name}")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset_name,
        dataset_root_dir=r".\datasets",
        batch_size=args.batch_size,
        num_workers=4)
    
    train(model_name=args.model_name,
          dataset_name=args.dataset_name,
          model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          epochs=args.epochs,
          lr=args.lr,
          weight_decay=args.weight_decay,
          device=device,
          pre_trained=args.pre_trained,
          momentum=0.9,
          save_dir="./model_ckpt"
    )
    
if __name__ == "__main__":
    main()


    



    
  



