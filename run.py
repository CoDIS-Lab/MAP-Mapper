# from Acolite import AcoliteProcessor
# from SemanticSegmentation import DatasetLoader
import datetime
import pandas as pd

from SentinelLoader.SentinelLoader import SentinelLoader

start = datetime.datetime.strptime("20210922", "%Y%m%d")
date_generated = pd.date_range(start, periods=10)
dates = []
date_generated.strftime("%Y%m%d")
for date in date_generated.strftime("%Y%m%d"):
    dates.append(str(date).replace("_", ""))
print(dates)

for i in range(len(dates)-1):
    start_date = dates[i]
    end_date = dates[i+1]
    SentinelLoader(start_date=start_date, end_date=end_date).run()
    print("d")
print("x")