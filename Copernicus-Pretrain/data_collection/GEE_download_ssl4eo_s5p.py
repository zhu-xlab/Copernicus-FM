import ee
#import numpy as np
#import glob
from multiprocessing.dummy import Lock, Pool
#import urllib
import os
#import re
import time
import requests
import csv
import random
import argparse
from datetime import date
import pdb


def filter_collection_s5p(collection,band_name,region,date1,date2):

    # Filter by total date range
    collection = collection.filterDate(date1, date2)
    # Filter by location
    #region = ee.Geometry.Point(center_coord).buffer(14000).bounds()
    collection = collection.filterBounds(region)

    def band_nan(image):
        mask = image.select(band_name).mask().Not().selfMask()
        stats = mask.reduceRegion(
        reducer=ee.Reducer.count(), 
        geometry=region,
        )
        return image.set('NAN_num', stats.get(band_name))
    
    collection = collection.map(band_nan)
    collection = collection.filter(ee.Filter.eq('NAN_num', 0))

    return collection


def download_image_s5p(collection,band_name,region,periods,save_root,point_id,center_coord):
    success_count = 0
    for period in periods:
        date1 = period[0]
        date2 = period[1]
        collection_m = collection.filterDate(date1, date2)
        #t1 = time.time()
        #collection_size = collection_m.size().getInfo()
        #t2 = time.time()
        #print('Time to get collection size:', t2-t1)
        #if collection_size < 1:
        #    #print('Not enough images for the sequence at location', center_coord, collection_size)
        #    continue
        try:
            image_mean = collection_m.mean().toFloat()
            url = image_mean.getDownloadUrl({
            'bands': [band_name], # 'H2O_column_number_density', 'cloud_height', 'sensor_altitude'
            'region': region,
            'scale': 1113.2, # GEE L3 resolution
            'format': 'GEO_TIFF'
            })
        except:
            continue
        #t3 = time.time()
        #print('Time to get download url:', t3-t2)
        str_idx = str(f"{point_id:07d}") 
        str_lon = str(f"{center_coord[0]:.2f}")
        str_lat = str(f"{center_coord[1]:.2f}")
        save_dir = os.path.join(save_root,'images',band_name,str_idx+'_'+str_lon+'_'+str_lat)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,str(date1)+'_'+str(date2)+'.tif')

        response = requests.get(url)
        with open(save_path, 'wb') as fd:
            fd.write(response.content)
        
        success_count += 1
    return success_count




class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self.lock = Lock()

    def update(self, delta: int = 1) -> int:
        with self.lock:
            self.value += delta
            return self.value
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir", type=str, default="./data/s5p", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, nargs="+",
        default=["COPERNICUS/S5P/OFFL/L3_CO","COPERNICUS/S5P/OFFL/L3_NO2","COPERNICUS/S5P/OFFL/L3_O3","COPERNICUS/S5P/OFFL/L3_SO2"], 
        help="GEE collection names"
    )
    parser.add_argument(
        "--band-name", type=str, nargs="+",
        default=["CO_column_number_density","tropospheric_NO2_column_number_density","O3_column_number_density","SO2_column_number_density"],
        help="GEE band names for corresponding collection"
    )
    # patch properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["2021-01-01","2021-02-01","2021-03-01","2021-04-01","2021-05-01","2021-06-01","2021-07-01","2021-08-01","2021-09-01","2021-10-01","2021-11-01","2021-12-01","2021-12-31"],
        help="reference dates",
    )
    parser.add_argument(
        "--radius", type=int, default=14000, help="patch radius in meters"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=12, help="number of images in sequence"
    )
    # download settings
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--log-freq", type=int, default=10, help="print frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    # sampler options
    parser.add_argument(
        "--match-file",
        type=str,
        required=True,
        help="match pre-sampled coordinates and indexes",
    )
    # number of locations to download
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 356232], # 0.25 degree grid joint with land mask
        help="indices to download",
    )
    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")


    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    #ee.Authenticate()
    ee.Initialize()

    # if resume
    ext_coords = {}
    ext_flags = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                key = int(row[0])
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
                ext_flags[key] = int(row[3])  # success or not
    else:
        ext_path = os.path.join(args.save_dir, "checked_locations.csv")
        with open(ext_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "lon", "lat", "CO", "NO2", "O3", "SO2"])

    # match from pre-sampled coords
    match_coords = {}
    with open(args.match_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            #key = int(row[0])
            key = int(row[3])
            val1 = float(row[1])
            val2 = float(row[2])
            match_coords[key] = (val1, val2)  # lon, lat

    start_time = time.time()
    counter = Counter()
    counter2 = Counter()


    # get collection
    collections = []
    for i in range(len(args.collection)):
        collection = ee.ImageCollection(args.collection[i]).select(args.band_name[i])
        collections.append(collection)

    #pdb.set_trace()

    # periods
    periods = []
    for i in range(0,len(args.dates)-1):
        periods.append([args.dates[i],args.dates[i+1]])

    #pdb.set_trace()

    def worker(idx):
        if idx in ext_coords.keys():
            return

        worker_start = time.time()

        center_coord = match_coords[idx]

        var_counts = [0,0,0,0]
        for i in range(len(collections)):
            collection = collections[i]
            band_name = args.band_name[i]
            region = ee.Geometry.Point(center_coord).buffer(args.radius).bounds()
            #pdb.set_trace()
            collection = filter_collection_s5p(collection,band_name,region,args.dates[0],args.dates[-1])
            success_count = download_image_s5p(collection,band_name,region,periods,args.save_dir,idx,center_coord)
            if success_count > 0:
                var_counts[i] += success_count

        count2 = counter2.update(1)
        #if out:
        #    count = counter.update(1)
        #    if count % args.log_freq == 0:
        #        print(f"Downloaded {count} locations in {time.time() - start_time:.3f}s.")
        if count2 % args.log_freq == 0:
            print(f"Checked {count2} locations in {time.time() - start_time:.3f}s.")

        # add to existing checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            success = var_counts
            data = [idx, *center_coord, *success]
            writer.writerow(data)

        # Throttle throughput to avoid exceeding GEE quota:
        # https://developers.google.com/earth-engine/guides/usage
        worker_end = time.time()
        elapsed = worker_end - worker_start
        num_workers = max(1, args.num_workers)
        time.sleep(max(0, num_workers / 100 - elapsed))

        return
    
    # set indices
    indices = list(range(args.indices_range[0], args.indices_range[1]))
    # indices should be within match_coords
    indices = [idx for idx in indices if idx in match_coords.keys()]

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        # parallelism data
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
