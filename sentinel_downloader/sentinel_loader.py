
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
import zipfile
from dotenv import load_dotenv
from utils.paths import base_path
from datetime import datetime
load_dotenv()

user_name = os.environ.get('USER_NAME')
password = os.environ.get('PASSWORD')


class SentinelLoader(object):

    def __init__(self, start_date, end_date, max_cloud_percentage, tile_id=None):
        # connect to api
        self.cloud_percentage = (0, max_cloud_percentage[0])
        self.download_path = None
        self.downloaded_files = None
        self.products = None
        self.api = SentinelAPI(user_name, password, 'https://scihub.copernicus.eu/dhus')
        # search by polygon
        self.footprint = geojson_to_wkt(read_geojson(os.path.join(base_path, "poly.geojson")))
        # for searching by time
        self.start_date = start_date
        self.end_date = end_date
        self.tile_id = tile_id

    # query api for matching products
    def get_product_data(self):

        query_kwargs = {
            'platformname': 'Sentinel-2',
            'producttype': 'S2MSI1C',
            'date': (self.start_date, self.end_date)}

        kw = query_kwargs.copy()
        if self.tile_id:
            kw['tileid'] = self.tile_id[0]
        kw['cloudcoverpercentage'] = self.cloud_percentage
        self.products = self.api.query(self.footprint, **kw)

        # check for duplicate tiles (these would cause errors when merging)
        # this code keeps the newest product (by generation date)
        product_dict = {}
        self_products_copy = self.products.copy()
        for product, value in self.products.items():
            tile_id = value["title"].split("_")[-2]
            # product discriminator should be the same for any tiles with multiple datatakes, excluding the last 6
            # numbers indicating time of generation product
            product_discriminator = value["title"].split("_")[-1].replace("T", "")

            datetime_object = datetime.strptime(product_discriminator, '%Y%m%d%H%M%S')

            try:
                # delete oldest product of the tile
                if datetime_object != product_dict[tile_id]:
                    del self_products_copy[product]

            except KeyError:
                product_dict[tile_id] = datetime_object
        self.products = self_products_copy



    # download all products
    # consider a path_filter for some bands?
    def download(self,  directory_path):
        self.api.download_all(self.products, directory_path)

    def get_download_list(self):
        self.downloaded_files = [f for f in os.listdir(self.download_path) if os.path.isfile(os.path.join(self.download_path, f))]
    
    def unzip_files(self):
        for file in self.downloaded_files:
            zip_path = os.path.join(self.download_path, file)
            with zipfile.ZipFile(os.path.join(zip_path), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)
                os.remove(zip_path)

    def run(self):
        self.get_product_data()
        if not self.products:
            print("No products found for " + self.start_date + " with max cloud cover percentage of " + str(self.cloud_percentage[1]))
        self.download_path = os.path.join(base_path, "data", "unprocessed")
        self.download(directory_path=self.download_path)
        self.get_download_list()
        self.unzip_files()


    

