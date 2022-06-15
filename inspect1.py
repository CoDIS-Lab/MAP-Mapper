import rasterio
import rasterio.mask
import os
from paths import base_path
import rasterio
from rasterio.enums import Resampling
path = os.path.join(base_path, "data", "unprocessed")
safe_files = os.listdir(path)



for safe_file in safe_files:
    tile_id = safe_file.split("_")[-2]
    date = safe_file.split("_")[-1][:8]
    out_img_file = f'{tile_id}_{date}_cloud.tif'
    out_dir = os.path.join(base_path, "data", "non_water_mask", out_img_file)
    safe_file_path = os.path.join(path, safe_file)
    upscale_factor = 2
    with rasterio.open(out_dir, "r") as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        out_meta = dataset.meta.copy()
        resample_img_file = f'{tile_id}_{date}_cloudresample.tif'
        resample_dir = os.path.join(base_path, "data", "non_water_mask", resample_img_file)
        out_meta.update({"height":data.shape[1], "width":data.shape[2], "transform":transform, "crs":"EPSG:32616"})
        with rasterio.open(resample_dir, "w", **out_meta) as dst:
            dst.write(data)

