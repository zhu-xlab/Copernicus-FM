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


def s3_mask_cloud(image):
    qa = image.select('quality_flags')
    brightBitMask = 1 << 27 # Bit 27 is the bright pixel
    mask = qa.bitwiseAnd(brightBitMask).eq(0)
    image.updateMask(mask)
    return image


def download_image_s3_olci(collection, point_id, center_coord, date1, date2, save_root, radius=14000, sequence_length=10, brightPixelsPercent=20):

    # filter nans
    def b4_nan(image):
        mask = image.select('Oa04_radiance').mask().Not().selfMask()
        stats = mask.reduceRegion(
        reducer=ee.Reducer.count(), 
        geometry=region,
        )
        return image.set('NAN_b4', stats.get('Oa04_radiance'))

    # filter by region and date
    region = ee.Geometry.Point(center_coord).buffer(radius).bounds()
    collection = collection.filterBounds(region)
    #collection = collection.filter(ee.Filter.contains('.geo', region))
    collection = collection.filterDate(date1, date2)
    # filter by cloud cover (bright pixels percent, except polar)
    if center_coord[1] > -50 and center_coord[1] < 60:
        collection = collection.filter(ee.Filter.lt('brightPixelsPercent', brightPixelsPercent)) # 20

    collection = collection.map(b4_nan)
    collection = collection.filter(ee.Filter.eq('NAN_b4',0))

    # get collection size, return if not enough images
    collection_size = collection.size().getInfo()
    if collection_size < sequence_length:
        print('Not enough images for the sequence at location', center_coord, collection_size)
        return None
    # get random images of sequence_length
    #collection = collection.randomColumn()
    #collection = collection.sort('random')
    #images_list = collection.toList(sequence_length)
    random_indices = random.sample(range(collection_size), sequence_length)
    images_list = collection.toList(collection_size)
    # download images
    for i in random_indices:
        image = ee.Image(images_list.get(i)).toFloat()
        gee_id = image.get('system:index').getInfo() 
        url = image.getDownloadUrl({
            'bands': ['Oa01_radiance','Oa02_radiance','Oa03_radiance','Oa04_radiance','Oa05_radiance','Oa06_radiance','Oa07_radiance',
                    'Oa08_radiance','Oa09_radiance','Oa10_radiance','Oa11_radiance','Oa12_radiance','Oa13_radiance','Oa14_radiance',
                    'Oa15_radiance','Oa16_radiance','Oa17_radiance','Oa18_radiance','Oa19_radiance','Oa20_radiance','Oa21_radiance'],
            'region': region,
            'scale': 300,
            'format': 'GEO_TIFF'
        })
        str_idx = str(f"{point_id:07d}") 
        str_lon = str(f"{center_coord[0]:.2f}")
        str_lat = str(f"{center_coord[1]:.2f}")
        save_dir = os.path.join(save_root,'images',str_idx+'_'+str_lon+'_'+str_lat)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,gee_id+'.tif')
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
        "--save-dir", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S3/OLCI", help="GEE collection name"
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=20, help="cloud percentage threshold"
    )
    # patch properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        # https://www.weather.gov/media/ind/seasons.pdf
        default=["2021-01-01", "2021-12-31"],
        help="reference dates",
    )
    parser.add_argument(
        "--radius", type=int, default=14000, help="patch radius in meters"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=10, help="number of images in sequence"
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

    # get collection
    collection = ee.ImageCollection(args.collection)

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
            writer.writerow(["id", "lon", "lat", "success"])

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

    def worker(idx):
        if idx in ext_coords.keys():
            return

        worker_start = time.time()

        center_coord = match_coords[idx]

        out = download_image_s3_olci(collection, idx, center_coord, args.dates[0], args.dates[1], save_root=args.save_dir, radius=args.radius, sequence_length=args.sequence_length, brightPixelsPercent=args.cloud_pct)
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