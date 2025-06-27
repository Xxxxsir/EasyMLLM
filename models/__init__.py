import sys
sys.path.append(r"C:\Users\Administrator\Desktop\JustinGao\Vscode Project\MachineLearning\EsayMLLM")  

      
from models.vit import vit_base  
from models.lenet import lenet        

model_dict = {
    'vit': vit_base,
    'lenet':lenet,
}

