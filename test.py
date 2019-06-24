from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
import os

# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    
    starting_num = 16;

    # Arguments
    parser.add_argument('--data-folder', type=str, default='../crop-faces-from-dataset/data_crop_1024_jpg_la60_68pt_sf40_ed40_mode_constant128',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default='./checkpoints/model.pt',
                        help='path of pre-trained weights (default: None)')

    args = parser.parse_args()
    args.device = torch.device("cpu")
    
    output_folder = (args.data_folder + '_masked')
    
    if not os.path.exists(output_folder):
    	os.makedirs(output_folder)

    image_files = sorted(glob('{}/*.jpg'.format(args.data_folder)))
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    print(len(image_files), ' files in folder ', args.data_folder)
    
    

#     fig = plt.figure()
    for i, image_file in enumerate(image_files):
        print(image_file)
        outname = image_file.replace(args.data_folder, output_folder)
        
        if(os.path.isfile(fname)):
        	print("mask exists, skipping")

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(args.device)

        # Forward Pass
        logits = model(torch_img)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        
        mask = mask.squeeze()
        mask[mask==1] = 128
        mask[mask==2] = 255
        mask[mask==3] = 55
        mask_224 = mask.astype(np.uint8)
        mask_1024 = cv2.resize(mask_224, (1024, 1024))
        kernel = np.ones((3,3), np.uint8) 
        
       
        
        cv2.imwrite(outname, mask_1024)
        
#         mask_384_eroded = cv2.erode(mask_1024,kernel,iterations = 3)
#         mask_384_dilated = cv2.dilate(mask_384_eroded,kernel ,iterations = 3)
        
#         mask_384_dilated_blur = cv2.blur(mask_384_dilated,(5,5))
#         mask_384_dilated_blur = cv2.blur(mask_384_dilated_blur,(7,7))
    
#         cv2.imshow('image', image)
#         cv2.imshow('mask', mask_1024)
#         cv2.waitKey(0)
        
        

#         Plot
#         ax = plt.subplot(2, args.batch_size, 2 * (i - starting_num) + 1)
#         ax.axis('off')
#         ax.imshow(image.squeeze())
# 
#         ax = plt.subplot(2, args.batch_size, 2 * (i - starting_num) + 2)
#         ax.axis('off')
#         ax.imshow(mask.squeeze())

#     plt.show()
