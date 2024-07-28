#Referrence: https://github.com/jeonsworld/ViT-pytorch/tree/main 


#Download Pretrain(relplace the MODEL_NAME with your pretrain):  wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz 

#quick start: pip install -r requirements.txt


#Train(replace the MODEL_NAME with your pretrain ) : python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type {MODEL_NAME} --pretrained_dir {MODEL_NAME}.npz --fp16 --fp16_opt_level O2


#e.g. If you want to use ViT-B_16.npz, then run:  python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir ViT-B_16.npz --fp16 --fp16_opt_level O2
