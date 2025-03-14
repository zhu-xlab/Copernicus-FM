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
from datetime import date, timedelta


def date2str(date: date) -> str:
    return date.strftime("%Y-%m-%d")


def get_period(date: date, days: int = 30):
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)
    #date3 = date1 - timedelta(days=365)
    #date4 = date2 - timedelta(days=365)
    #date5 = date1 - timedelta(days=365*2)
    #date6 = date2 - timedelta(days=365*2)
    return (
        date2str(date1),
        date2str(date2),
        #date2str(date3),
        #date2str(date4),
        #date2str(date5),
        #date2str(date6),
    )  # three-years buffer


def filter_collection_s1(collection,region):

    # Filter by location
    #region = ee.Geometry.Point(center_coord).buffer(14000).bounds()
    collection = collection.filterBounds(region)
    collection = collection.filter(ee.Filter.contains('.geo', region))

    return collection


def download_image_s1(collection,region,periods,save_root,point_id,center_coord,grid_coord):
    
    urls = []
    gee_ids = []
    for period in periods:
        collection_m = collection.filter(
            ee.Filter.date(period[0], period[1])
            #ee.Filter.Or(
                #ee.Filter.date(period[0], period[1]),
                #ee.Filter.date(period[2], period[3]),
                #ee.Filter.date(period[4], period[5]),
            #)
        )
        try:
            image = collection_m.first().toFloat()
            url = image.getDownloadUrl({
            'bands': ['HH','HV'], # S1 EW mode
            'region': region,
            'scale': 10, # GEE S1 resolution
            'format': 'GEO_TIFF'
            })
            gee_id = image.get('system:index').getInfo()
            urls.append(url)
            gee_ids.append(gee_id)
        except:
            return 0


    str_grid_id = str(f"{grid_coord[0]:07d}")
    str_grid_lon = str(f"{grid_coord[1]:.2f}")
    str_grid_lat = str(f"{grid_coord[2]:.2f}")        
    str_idx = str(f"{point_id:07d}") 
    str_lon = str(f"{center_coord[0]:.2f}")
    str_lat = str(f"{center_coord[1]:.2f}")
    save_dir = os.path.join(save_root,'images',str_grid_id+'_'+str_grid_lon+'_'+str_grid_lat,str_idx+'_'+str_lon+'_'+str_lat)
    os.makedirs(save_dir, exist_ok=True)
        
    for url,gee_id in zip(urls,gee_ids):
        save_path = os.path.join(save_dir,gee_id+'.tif')
        response = requests.get(url)
        with open(save_path, 'wb') as fd:
            fd.write(response.content)
        
    return 1


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
        "--save-dir", type=str, default="./data/s1", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S1_GRD", help="GEE collection names"
    )
    # patch properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["2022-12-21", "2022-09-22", "2022-06-21", "2022-03-20"],
        help="reference dates",
    )
    parser.add_argument(
        "--radius", type=int, default=1320, help="patch radius in meters"
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
        default=[251079, 1000000], # ssl4eo-s12-plus ids
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
            writer.writerow(["s2_index", "lon", "lat", "grid_index", "grid_lon", "grid_lat", "success"])

    # match from pre-sampled coords
    match_coords = {}
    grid_coords = {}
    with open(args.match_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            key = int(row[0])
            val1 = float(row[1])
            val2 = float(row[2])
            match_coords[key] = (val1, val2)  # lon, lat
            grid_idx = int(row[3])
            grid_lon = float(row[4])
            grid_lat = float(row[5])
            grid_coords[key] = (grid_idx,grid_lon,grid_lat) # ssl4eo-s12-plus ids to grid ids

    start_time = time.time()
    counter = Counter()
    counter2 = Counter()


    # get collection
    collection_s1 = ee.ImageCollection(args.collection)

    # periods
    dates = []
    for d in args.dates:
        dates.append(date.fromisoformat(d))
    periods = [get_period(date, days=360) for date in dates]


    #pdb.set_trace()
    def worker(idx):
        if idx in ext_coords.keys():
            return

        worker_start = time.time()

        center_coord = match_coords[idx]
        grid_coord = grid_coords[idx]

        region = ee.Geometry.Point(center_coord).buffer(args.radius).bounds()
        collection = collection_s1.filterBounds(region)
        collection = collection.filter(ee.Filter.contains('.geo', region))
        success_count = download_image_s1(collection,region,periods,args.save_dir,idx,center_coord,grid_coord)

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
            success = success_count
            data = [idx, *center_coord, *grid_coord, success]
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