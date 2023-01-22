import os
import sys
import random
import rasterio
import numpy as np
from os.path import dirname as up
import torch
from semantic_segmentation.Unet import UNet
from utils.dir_management import base_path
import copy
from semantic_segmentation.smooth_patches.smooth_tiled_predictions import predict_img_with_smooth_windowing
import torchvision.transforms as transforms

# for bands 4, 6, 8, 11
bands_mean = np.array(np.array([0.03163572, 0.03457443, 0.03436435, 0.02358126]).astype('float32'))
bands_std = np.array([0.04967381, 0.06458357, 0.07120246, 0.05111466]).astype('float32')


sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def predict(model, image):
    predictions = model(torch.movedim(image, (0, 3, 1, 2), (0, 1, 2, 3)))
    probs = torch.nn.functional.softmax(predictions.detach(), dim=1).cpu().numpy()
    return np.moveaxis(probs, (0, 3, 1, 2), (0, 1, 2, 3))


def create_image_prediction():
    print("making smoothly blended predictions....")

    options = {"input_channels": 4,
               "output_channels": 2,
               "model_path": os.path.join(base_path, 'semantic_segmentation', 'unet', 'trained_models', 'model.pth'),
               "gen_masks_path": os.path.join(base_path, 'data', 'predicted_patches')}

    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_bands=options['input_channels'],
                 output_classes=options['output_channels'],)

    model.to(device)

    # Load model
    model_file = options['model_path']
    model.load_state_dict(torch.load(model_file, map_location=device))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    patch_path = os.path.join(base_path, "data", "patches")
    for file in os.listdir(patch_path):
        with rasterio.open(os.path.join(patch_path, file)) as src:
            input_img = src.read()
            # this code enables images patches with all NaN values to be skipped from predictions
            if np.isnan(input_img).all():
                print(f"{file} contains all NaN values, skipping...")
                out_meta = src.meta
                input_img = input_img[0, :, :]
                input_img = np.expand_dims(input_img, axis=0)
                out_meta.update(
                    {"count": 1,
                     "dtype": "float32",
                     "nodata": 99})
                with rasterio.open(
                        os.path.join(base_path, "data", "predicted_patches", file.strip(".tif") + "_probs.tif"), "w",
                        **out_meta) as dst:
                    dst.write(input_img)
            # prepare and preprocess image patch prior to patch predictions
            else:
                print(f"making predictions on {file}")
                input_img = np.moveaxis(input_img, (1, 2, 0), (0, 1, 2))
                out_meta = src.meta
                input_img = transform_test(input_img)
                input_img = standardization(input_img)
                input_img = torch.movedim(input_img, (1, 2, 0), (0, 1, 2))
                # send to device
                input_img = input_img.to(device)
                with torch.no_grad():
                    predictions_smooth = predict_img_with_smooth_windowing(
                        input_img,
                        window_size=32,
                        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                        nb_classes=2,
                        model=model,
                        pred_func=lambda img_batch_subdiv: predict(model, torch.from_numpy(img_batch_subdiv))
                    )
                    probs = copy.deepcopy(predictions_smooth)
                    probs = probs[:, :, 0]
                    probs = np.expand_dims(probs, axis=0)
                    out_meta_probs = out_meta
                    out_meta_probs.update(
                        {"height": probs.shape[1],
                         "width": probs.shape[2],
                         "count": 1,
                         "dtype": "float32",
                         "nodata": 99})
                    print("predicted_shape: ",  probs.shape)
                    with rasterio.open(
                            os.path.join(base_path, "data", "predicted_patches", file.strip(".tif") + "_probs.tif"), "w",
                            **out_meta_probs) as dst:
                        dst.write(probs)


