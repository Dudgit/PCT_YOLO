#!/usr/bin/python3

import numpy as np
from random import randint, gauss, Random
import json
import os
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing as mp
# import xml.etree.ElementTree as ET
from lxml import etree as ET
import pathlib

xsize = 600
ysize = 600
mu = 4
sigma = 3
num = 10000
maxHits = 20
imFolder = "images"
anFolder = "annotations"
dtFolder = "data"
listName = "list"


def getHit(local_random):

    x = local_random.randint(0, xsize)
    y = local_random.randint(0, ysize)
    r = round(abs(local_random.gauss(mu=mu, sigma=sigma)), 2)

    xmin = x-r
    ymin = y-r
    xmax = x+r
    ymax = x+r

    return (x, y, r), (xmin, ymin, xmax, ymax)


def getAnnots(fName, imDims, boxes):

    annotation = ET.Element('annotation')
    annotation.set('verified', 'yes')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = imFolder

    filename = ET.SubElement(annotation, 'filename')
    filename.text ="hit_" +str(fName)+".png"

    path = ET.SubElement(annotation, 'path')
    path.text = str(pathlib.Path().absolute())

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Generator script'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(imDims[0])
    height = ET.SubElement(size, 'height')
    height.text = str(imDims[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(imDims[2])

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = "Hit"
        pose = ET.SubElement(obj, 'pose')
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = "0"
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = "0"
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(box[0])
        ymin.text = str(box[1])
        xmax.text = str(box[2])
        ymax.text = str(box[3])

    tree = ET.ElementTree(annotation)
    tree.write(anFolder+"/"+fName+".xml",
               pretty_print=True)
    # print("Save xml to %s done" % oFile)


def drawImage(num):

    fileName = "{:06d}".format(num)

    # Create a 600x600x3 array of 8 bit unsigned integers; make them white
    data = np.full((xsize, ysize, 3), (255, 255, 255), dtype=np.uint8)

    x = []
    y = []
    r = []
    boxes = []

    local_random = Random()
    local_random.seed(num)

    hits = local_random.randint(1, maxHits)

    for h in range(hits):
        hit, box = getHit(local_random)
        x.append(hit[0])
        y.append(hit[1])
        r.append(hit[2])
        boxes.append(box)

        for i in range(xsize):
            for j in range(ysize):
                if (i-x[h])**2+(j-y[h])**2 <= r[h]**2:
                    data[i, j] = [0, 0, 0]

    img = Image.fromarray(data)
    img.save(imFolder+"/"+"hit_"+fileName+".png")

    imgData = {
        "num": fileName,
        "hits": hits,
        "x": x,
        "y": y,
        "r": r
    }

    getAnnots(fileName, (xsize, ysize, 3), boxes)

    with open(dtFolder+"/"+listName+".json", mode='ab+') as handle:
        handle.seek(-1, os.SEEK_END)
        handle.truncate()

    with open(dtFolder+"/"+listName+".json", mode='a+', encoding='utf-8') as handle:
        if fileName != "000000":
            handle.write(',')
        json.dump(imgData, handle, indent=2)
        handle.write(']')


def main():

    with open(dtFolder+"/"+listName+".json", mode='w', encoding='utf-8') as handle:
        empty = []
        json.dump(empty, handle)
        print("file "+dtFolder+"/"+listName+".json created")

    # Comment the following if you do not want run in CPU parallel mode.
    num_cores = 1
    num_cores = mp.cpu_count()

    drawImage(0)

    Parallel(n_jobs=num_cores)(delayed(drawImage)(i) for i in range(1, num))


main()
