import torchvision.models as models
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from PIL import Image
import torch
import os
from pathlib import Path
from tqdm import tqdm


def get_image_files(root,img_extensions=['.JPEG','.jpg','.png']):
    """get image paths from root folder"""
    img_paths=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_path=Path(os.path.join(path, name))
            if file_path.suffix  in img_extensions:
                img_paths.append(file_path)
    return img_paths


def get_img_training_data(root,bs=128,feature_extractor_fn=models.resnet18,
                         pooling_layer='avgpool',device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):
    """for a given root folder of images (split by class folder )
    return features and labels for all images from all classes created by a 
    pretrained model defined by feature extractor fn
    """
    img_paths=get_image_files(root)
    transform = T.Compose([T.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
                           ,T.Resize(256), T.CenterCrop(224)])
    
    ## instantiate  pretrained model and put it in  eval mode : 
    model=feature_extractor_fn(pretrained=True)
    feature_ex =create_feature_extractor(model, return_nodes=[pooling_layer])
    feature_ex.eval()
    feature_ex.to(device)
    
    
    features,lbls=[],[]
   # while len(img_paths>0):
    for batch_num in tqdm(range(len(img_paths)//bs)):
        bs=min(bs,len(img_paths))
        batch_imgs=img_paths[:bs]
        yb=[]
        img_tensor_list=[]
        for img_path in batch_imgs:
            img_tensor= T.functional.to_tensor(Image.open(img_path))
           ##append rgb 3 channel images. print the rest
            if img_tensor.shape[0]==3:
                img_tensor=transform(img_tensor)
                img_tensor_list.append(img_tensor)
                yb.append(img_path.parent.stem)
            else:
                print(img_path)
        xb=torch.stack(img_tensor_list)
        with torch.no_grad():
            #put data on same device as the model
            xb=xb.to(device)
            fb=feature_ex(xb)['avgpool'].squeeze(3).squeeze(2).cpu().numpy()
        features.append(fb)
        lbls.append(np.array(yb))
        img_paths=img_paths[bs:]
     
    return np.concatenate(features),np.concatenate(lbls)
        
  