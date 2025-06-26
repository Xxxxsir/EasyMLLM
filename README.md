当前支持

🏁 模型：
-ViT (vit_base)

🩺 数据集：
-COVID_19_Radiography_Dataset

项目结构：
EsayMLLM/

├─ models/

├─ datasets/

├─ results/

├─ datasets.py

├─ inference.py

├─ train.py

├─ utils.py

👇 TODO
-增加更多模型支持

-增加更多数据集

-增加混合精度训练

-增加可视化脚本

快速开始：

模型训练
```
python train.py \
    --model_name vit \
    --dataset_name covid \
    --classes_num 4 \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --seed 42 \
    --pre_trained False
```

推理
```
python inference.py 
    --model_name vit \
    --checkpoint_path path_to_pretrained_model \
    --image_path path_to_img \
    --dataset covid \
    --num_classes 4
```


