
import os
import sys
import random
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up
import torch
import torchvision.transforms as transforms
from SemanticSegmentation.Unet import UNet
from paths import base_path
bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main(options):
    # Transformations
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = UNet(input_bands = options['input_channels'], 
                 output_classes = options['output_channels'], 
                 hidden_channels = options['hidden_channels'])

    model.to(device)

    # Load model
    model_file = options['model_path']
    model.load_state_dict(torch.load(model_file, map_location = device))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():
                
        path = os.path.join(base_path, "data", "patches")
        ROIs = [name for name in os.listdir(path)]
        impute_nan = np.tile(bands_mean, (256,256,1))
                    
        for roi in tqdm(ROIs):
            roi_file = os.path.join(path, roi)     # Get File path
        
            os.makedirs(options['gen_masks_path'], exist_ok=True)
        
            output_image = os.path.join(options['gen_masks_path'], os.path.basename(roi_file).split('.tif')[0] + '_unet.tif')

            # Read metadata of the initial image
            with rasterio.open(roi_file, mode ='r') as src:
                tags = src.tags().copy()
                meta = src.meta
                image = src.read()
                image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                dtype = src.read(1).dtype
        
            # Update meta to reflect the number of layers
            meta.update(count = 1)
        
            # Write it
            with rasterio.open(output_image, 'w', **meta) as dst:
                nan_mask = np.isnan(image)
                image[nan_mask] = impute_nan[nan_mask]
                image = transform_test(image)
                image = standardization(image)
                # Image to Cuda if exist
                image = image.to(device)
                # Predictions
                logits = model(image.unsqueeze(0))
                probs = torch.nn.functional.softmax(logits.detach(), dim=1).cpu().numpy()
                probs = probs.argmax(1).squeeze()+1
                # Write the mask with georeference
                dst.write_band(1, probs.astype(dtype).copy()) # In order to be in the same dtype
                dst.update_tags(**tags)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Unet parameters
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features')
    
    # Unet model path
    parser.add_argument('--model_path', default=os.path.join(base_path, 'SemanticSegmentation', 'unet', 'trained_models', 'model.pth'), help='Path to Unet pytorch model')
    
    parser.add_argument('--gen_masks_path', default=os.path.join(base_path, 'data', 'predicted_unet'), help='Path to where to produce store predictions')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)