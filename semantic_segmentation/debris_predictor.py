import os
import sys
import random
import numpy
import rasterio
import numpy as np
from os.path import dirname as up
import torch
import torchvision.transforms as transforms
from semantic_segmentation.Unet import UNet
from paths import base_path
from smooth_patches.smooth_tiled_predictions import predict_img_with_smooth_windowing

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
                      0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
                       0.03875053, 0.03436435, 0.0392113, 0.02358126, 0.01588816]).astype('float32')

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# prepare subdiv for model predictions
def image_to_neural_input(subdiv):
    return torch.movedim(torch.from_numpy(subdiv), (0, 3, 1, 2), (0, 1, 2, 3))


def predict(model, image):
    predictions = model(torch.movedim(image, (0, 3, 1, 2), (0, 1, 2, 3)))
    probs = torch.nn.functional.softmax(predictions.detach(), dim=1).cpu().numpy()
    return np.moveaxis(probs, (0, 3, 1, 2), (0, 1, 2, 3))


def predict_with_smooth_blending():
    print("making smoothly blended predictions....")

    options = {"input_channels": 11,
               "output_channels": 11,
               "hidden_channels": 16,
               "model_path": os.path.join(base_path, 'semantic_segmentation', 'unet', 'trained_models', 'model.pth'),
               "gen_masks_path": os.path.join(base_path, 'data', 'predicted_unet')}

    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_bands=options['input_channels'],
                 output_classes=options['output_channels'],
                 hidden_channels=options['hidden_channels'])

    model.to(device)

    # Load model
    model_file = options['model_path']
    model.load_state_dict(torch.load(model_file, map_location=device))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    patch_path = os.path.join(base_path, "data", "large_patches")
    for file in os.listdir(patch_path):
        print("predictions on " + file)
        with rasterio.open(os.path.join(patch_path, file)) as src:
            # prepare and preprocess image patch prior to patch predictions
            input_img = src.read()
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
                    window_size=256,
                    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                    nb_classes=11,
                    model=model,
                    pred_func=lambda img_batch_subdiv: predict(model, torch.from_numpy(img_batch_subdiv))
                )
                # makes nan values max value in cloud channel
                # nan values are classified as cloud (ignored in final analysis)
                numpy.nan_to_num(predictions_smooth[:, :, 5], copy=False, nan=99)
                # uses the maximum value of the predictions channels to classify each pixel,
                argmax = np.nanargmax(predictions_smooth, axis=2)
                # increase each value of prediction mask by 1 (0 - 10 becomes 1 - 11)
                argmax = argmax + 1
                final_prediction = np.expand_dims(argmax, axis=0)
                final_prediction = final_prediction.astype(np.uint8)

                out_meta.update(
                    {"height": final_prediction.shape[1],
                     "width": final_prediction.shape[2],
                     "count": 1,
                     "dtype": "uint8",
                     "nodata": 255})
                with rasterio.open(
                        os.path.join(base_path, "data", "predicted_unet", file.strip(".tif") + "_predict.tif"), "w",
                        **out_meta) as dst:
                    dst.write(final_prediction)
