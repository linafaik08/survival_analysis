# Source code:
# https://github.com/chrise96/image-to-coco-json-converter/blob/master/src/create_annotations.py

from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import os
import json
import pandas as pd

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors

    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image

                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, tol, threshold_area):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)

        poly = poly.simplify(tol, preserve_topology=True)

        if(poly.is_empty) or poly.area<threshold_area:
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def images_annotations_info(paths_df, category_colors, multipolygon_ids, n_images_max=None, tol=100, threshold_area = 10):
    annotation_id = 0
    annotations = []
    images = []

    n_images = paths_df.shape[0]

    for image_id in range(n_images):
        line = paths_df.iloc[image_id]

        print('image {}/{}'.format(image_id+1, n_images))

        mask_image_open = Image.open(line.path_mask).convert("RGB")
        print(line.path_mask)

        w, h = mask_image_open.size

        # "images" info
        image = create_image_annotation(line.path_image, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)

        for color, sub_mask in sub_masks.items():

            if color in category_colors.keys():

                category_id = category_colors[color]

                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask,  tol, threshold_area)

                #return sub_masks, polygons, segmentations

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:

                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)

                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1

        image_id += 1

        if n_images_max is not None and n_images_max<=image_id:
            break

    return images, annotations, annotation_id