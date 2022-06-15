
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
import zipfile
from dotenv import load_dotenv
from paths import base_path
load_dotenv()

user_name = os.environ.get('USER_NAME')
password = os.environ.get('PASSWORD')


class SentinelLoader(object):

    def __init__(self, start_date="20210918", end_date="20210919"):
        # connect to api
        self.download_path = None
        self.downloaded_files = None
        self.products = None
        self.api = SentinelAPI(user_name, password, 'https://scihub.copernicus.eu/dhus')
        # search by polygon
        print(base_path)
        self.footprint = geojson_to_wkt(read_geojson(os.path.join(base_path, "poly.geojson")))
        # for searching by time
        self.start_date = start_date
        self.end_date = end_date

    # query api for matching products
    def get_product_data(self):
        self.products = self.api.query(self.footprint, date=(self.start_date, self.end_date), platformname='Sentinel-2', producttype='S2MSI1C')

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
        print(self.products)
        self.download_path = os.path.join(base_path, "data", "unprocessed")
        self.download(directory_path=self.download_path)
        self.get_download_list()
        self.unzip_files()

    
if __name__ == '__main__':
    SentinelLoader().run()

    

