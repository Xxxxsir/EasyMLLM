import sys
sys.path.append(r"C:\Users\Administrator\Desktop\JustinGao\Vscode Project\MachineLearning\EsayMLLM")  

      
from models.vit import vit_base_patch16_224            

model_dict = {
    'vit': vit_base_patch16_224
}

def create_model(model_name, num_classes,pre_trained:bool=False,pretrained_path:str=None, **kwargs):
    return model_dict[model_name](num_classes=num_classes,pre_trained=pre_trained, pretrained_path=pretrained_path)
