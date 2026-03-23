import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from TDS_net import TDSNet

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "./ckpts/snake-model_29.pth"

    img_path = "test_samples/5.jpg"
    name = img_path.split('/')[-1]
    print('name:', name)

    mask_name = name.split('.')[0] + '.png'

    #mask_name = "crack_mask-IMG13-3rs"
    classes = 1  # exclude background

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TDSNet(n_classes=classes+1)
    # model = RTFormerSlim(num_classes=classes+1)
    # model = SegNeXt_S(num_classes=classes+1)

    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"], strict=False)
    else:
        model.load_state_dict(weights, strict=False)
    model.to(device)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(512),  # 将图像的短边调整为288，并保持图像纵横比不变。  若有需要可以不注释这一行
        transforms.Normalize(mean=(0.590, 0.578, 0.561),
                             std=(0.099, 0.099, 0.099))
    ])
    #transforms.Normalize(mean=(0.590, 0.578, 0.561),
    #                     std=(0.099, 0.099, 0.099))
    #mean = (0.37113734, 0.33957041, 0.29494928)
    #std = (0.05629448, 0.05534535, 0.052346)

    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))     #正常预测
        #output,edge = model(img.to(device))      #带边缘的预测

        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        mask = Image.fromarray(prediction)
        mask.save("./test_res/29/" + 'crak_mask-'+mask_name )
        print(mask_name + ' is finished')


if __name__ == '__main__':
    main()
