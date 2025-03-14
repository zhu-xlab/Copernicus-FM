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


def download_image_dem(collection, center_coord, radius, point_id, save_root):
    region = ee.Geometry.Point(center_coord).buffer(radius).bounds()
    collection = collection.filterBounds(region)

    try:
        image = collection.mean()
        url = image.getDownloadUrl({
        'bands': ['DEM'],
        'region': region,
        'scale': 30,
        'format': 'GEO_TIFF'
        })
    except:
        print('No DEM available for location', center_coord)
        return None

    str_idx = str(f"{point_id:07d}") 
    str_lon = str(f"{center_coord[0]:.2f}")
    str_lat = str(f"{center_coord[1]:.2f}")
    save_dir = os.path.join(save_root,'images',str_idx+'_'+str_lon+'_'+str_lat)    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,'dem.tif')

    response = requests.get(url)
    with open(save_path, 'wb') as fd:
        fd.write(response.content)
    
    return save_path


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
        "--save-dir", type=str, default="./data/dem", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/DEM/GLO30", help="GEE collection names"
    )
    # patch properties
    parser.add_argument(
        "--radius", type=int, default=14000, help="patch radius in meters"
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
            writer.writerow(["index", "lon", "lat", "success"])

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
    collection = ee.ImageCollection(args.collection)


    def worker(idx):
        if idx in ext_coords.keys():
            return

        worker_start = time.time()

        center_coord = match_coords[idx]

        out = download_image_dem(collection, center_coord, args.radius, idx, args.save_dir)

        count2 = counter2.update(1)
        if out:
           count = counter.update(1)
           if count % args.log_freq == 0:
               print(f"Downloaded {count} locations in {time.time() - start_time:.3f}s.")
        if count2 % args.log_freq == 0:
            print(f"Checked {count2} locations in {time.time() - start_time:.3f}s.")

        # add to existing checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            if out:
                success = 1
            else:
                success = 0
            data = [idx, *center_coord, success]
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