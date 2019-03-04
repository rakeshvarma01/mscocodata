"""
Copyright 2018-2019 University of Waikato, Hamilton, NZ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import argparse
import os
import traceback
import logging
import datetime
import json
from collections import OrderedDict
import PIL.Image as pil
from adams2detectron.report import read_objects, determine_labels, fix_labels
from adams2detectron.report import SUFFIX_TYPE, SUFFIX_X, SUFFIX_Y, SUFFIX_WIDTH, SUFFIX_HEIGHT, REPORT_EXT
from adams2detectron.report import SUFFIX_POLY_X, SUFFIX_POLY_Y, PREFIX_OBJECT, DEFAULT_LABEL

# logging setup
logging.basicConfig()
logger = logging.getLogger("adams2detectron.convert")
logger.setLevel(logging.INFO)


def image_for_report(report_file):
    """
    Determines the file name for the report file.

    :param report_file: the report file to get the associated image for
    :type report_file: str
    :return: the image file_name, None if not present
    :rtype: str
    """
    jpg_lower = report_file.replace(REPORT_EXT, ".jpg")
    jpg_upper = report_file.replace(REPORT_EXT, ".JPG")
    png_lower = report_file.replace(REPORT_EXT, ".png")
    png_upper = report_file.replace(REPORT_EXT, ".PNG")
    if os.path.exists(jpg_lower):
        return jpg_lower
    elif os.path.exists(jpg_upper):
        return jpg_upper
    elif os.path.exists(png_lower):
        return png_lower
    elif os.path.exists(png_upper):
        return png_upper
    else:
        return None


def add_images(images, report_files, verbose=False):
    """
    Adds all the images to the images list.

    :param images: the list to append to
    :type images: list
    :param report_files: the list of report files (strings) to use for determining the images
    :type report_files: list
    :param verbose: whether to be verbose in the output
    :type verbose: bool
    """
    captured = datetime.datetime.now().isoformat()
    for img_id, report_file in enumerate(report_files, start=1):
        img_file = image_for_report(report_file)
        if img_file is None:
            continue
        if verbose:
            logger.info('Processing image: %s' % img_file)
        img = pil.open(img_file)
        data = OrderedDict()
        data['id'] = img_id
        data['width'] = img.size[0]
        data['height'] = img.size[1]
        data['file_name'] = os.path.basename(img_file)
        data['license'] = 1
        data['flickr_url'] = ''
        data['coco_url'] = ''
        data['date_captured'] = captured
        images.append(data)


def add_annotations(annotations, report_files, mappings=None, labels=None, polygon=False, verbose=False):
    """
    Adds all the objects to the annotations list.

    :param annotations: the list to append to
    :type annotations: list
    :param report_files: the list of report files (strings) to use for determining the images
    :type report_files: list
    :param mappings: the label mappings for replacing labels (key: old label, value: new label)
    :type mappings: dict
    :param labels: dictionary of label names as keys and category IDs as value
    :type labels: dict
    :param polygon: whether to use polygon information instead of bounding box for output
    :type polygon: bool
    :param verbose: whether to be verbose in the output
    :type verbose: bool
    """
    obj_id = 0
    for img_id, report_file in enumerate(report_files, start=1):
        img_file = image_for_report(report_file)
        if img_file is None:
            continue
        objects = read_objects(report_file, verbose=verbose)
        if mappings is not None:
            fix_labels(objects, mappings)
        for obj_key in objects:
            obj_id += 1
            obj = objects[obj_key]
            if SUFFIX_TYPE not in obj:
                obj[SUFFIX_TYPE] = DEFAULT_LABEL
            if obj[SUFFIX_TYPE] in labels:
                data = OrderedDict()
                data['id'] = obj_id
                data['image_id'] = img_id
                data['category_id'] = labels[obj[SUFFIX_TYPE]]
                if polygon and (SUFFIX_POLY_X in obj) and (SUFFIX_POLY_Y in obj):
                    poly_x = [int(x) for x in obj[SUFFIX_POLY_X].split(',')]
                    poly_y = [int(y) for y in obj[SUFFIX_POLY_Y].split(',')]
                    xmin = min(poly_x)
                    xmax = max(poly_x)
                    ymin = min(poly_y)
                    ymax = max(poly_y)
                    data['segmentation'] = list()
                    for i in range(len(poly_x)):
                        data['segmentation'].append(poly_x[i])
                        data['segmentation'].append(poly_y[i])
                    data['segmentation'] = [data['segmentation']]
                else:
                    xmin = obj[SUFFIX_X]
                    xmax = obj[SUFFIX_X] + obj[SUFFIX_WIDTH] - 1
                    ymin = obj[SUFFIX_Y]
                    ymax = obj[SUFFIX_Y] + obj[SUFFIX_HEIGHT] - 1
                    data['segmentation'] = [[
                        xmin, ymin,
                        xmax, ymin,
                        xmax, ymax,
                        xmin, ymax,
                ]]
                width = xmax - xmin + 1
                height = ymax - ymin + 1
                data['area'] = float(width * height)
                data['bbox'] = [xmin, ymin, width, height]
                data['iscrowd'] = 0
                annotations.append(data)


def convert(input_dir, input_files, output_file, mappings=None, regexp=None, labels=None, polygon=False,
            pretty_print=False, verbose=False):
    """
    Converts the images and annotations (.report) files into MS COCO JSON.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param input_files: the file containing the report files to use
    :type input_files: str
    :param output_file: the output file for MS COCO JSON
    :type output_file: str
    :param mappings: the label mappings for replacing labels (key: old label, value: new label)
    :type mappings: dict
    :param regexp: the regular expression to use for limiting the labels stored
    :type regexp: str
    :param labels: the predefined list of labels to use
    :type labels: list
    :param polygon: whether to use polygon information instead of bounding box for output
    :type polygon: bool
    :param pretty_print: whether to generate pretty-printed JSON output
    :type pretty_print: bool
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    """

    if labels is None:
        labels = determine_labels(input_dir=input_dir, input_files=input_files, mappings=mappings,
                                  regexp=regexp, verbose=verbose)
    label_indices = dict()
    for i, l in enumerate(labels):
        label_indices[l] = i+1

    if verbose:
        logging.info("labels considered: %s", labels)

    # determine files
    if input_dir is not None:
        report_files = list()
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(REPORT_EXT):
                    report_files.append(os.path.join(input_dir, subdir, f))
    else:
        report_files = input_files[:]

    # compile json
    coco = OrderedDict()

    info = OrderedDict()
    info['year'] = datetime.date.today().year
    info['version'] = str(datetime.datetime.now())
    info['description'] = 'ADAMS annotations'
    info['contributor'] = 'ADAMS'
    info['url'] = 'https://adams.cms.waikato.ac.nz/'
    info['date_created'] = datetime.datetime.now().isoformat()
    coco['info'] = info

    license = OrderedDict()
    license['id'] = 1
    license['name'] = 'proprietary'
    license['url'] = ''
    coco['licenses'] = list()
    coco['licenses'].append(license)

    coco['categories'] = list()
    for l in label_indices:
        cat = OrderedDict()
        cat['id'] = label_indices[l]
        cat['name'] = l
        cat['supercategory'] = 'shape'
        coco['categories'].append(cat)
    coco['images'] = list()
    coco['annotations'] = list()
    add_images(coco['images'], report_files, verbose=verbose)
    add_annotations(coco['annotations'], report_files, mappings=mappings, labels=label_indices, polygon=polygon, verbose=verbose)

    # save to file
    with open(output_file, 'w') as outfile:
        if pretty_print:
            json.dump(coco, outfile, indent=2)
        else:
            json.dump(coco, outfile)


def main(args):
    """
    Runs the conversion from command-line. Use -h/--help to see all options.

    :param args: the command-line arguments to parse
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description='Converts ADAMS annotations (image and .report files) into MS COCO JSON for the '
                    + 'Detectron framework.\n'
                    + 'Assumes "' + PREFIX_OBJECT + '" as prefix and "' + SUFFIX_TYPE + '" for the label. '
                    + 'If no "' + SUFFIX_TYPE + '" present, the generic label "' + DEFAULT_LABEL + '" will '
                    + 'be used instead.')
    parser.add_argument(
        "-i", "--input", metavar="dir_or_file", dest="input", required=True,
        help="input directory with report files or text file with one absolute report file name per line")
    parser.add_argument(
        "-o", "--output", metavar="file", dest="output", required=True,
        help="name of output file for JSON")
    parser.add_argument(
        "-m", "--mapping", metavar="old=new", dest="mapping", action='append', type=str, required=False,
        help="mapping for labels, for replacing one label string with another (eg when fixing/collapsing labels)", default=list())
    parser.add_argument(
        "-r", "--regexp", metavar="regexp", dest="regexp", required=False,
        help="regular expression for using only a subset of labels", default="")
    parser.add_argument(
        "-l", "--labels", metavar="label1,label2,...", dest="labels", required=False,
        help="comma-separated list of labels to use", default="")
    parser.add_argument(
        "-g", "--polygon", action="store_true", dest="polygon", required=False,
        help="output polygon (if available) instead of bounding box")
    parser.add_argument(
        "-p", "--pretty_print", action="store_true", dest="pretty_print", required=False,
        help="whether to generate pretty-printed JSON")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", required=False,
        help="whether to be more verbose when generating the records")
    parsed = parser.parse_args(args=args)

    # checks
    if not os.path.exists(parsed.input):
        raise IOError("Input does not exist:", parsed.input)
    if os.path.isdir(parsed.output):
        raise IOError("Output is a directory:", parsed.output)

    # interpret input (dir or file with report file names?)
    if os.path.isdir(parsed.input):
        input_dir = parsed.input
        input_files = None
    else:
        input_dir = None
        input_files = list()
        with open(parsed.input) as fp:
            for line in fp:
                input_files.append(line.strip())

    # generate label mappings
    mappings = None
    if len(parsed.mapping) > 0:
        mappings = dict()
        for m in parsed.mapping:
            old, new = m.split("=")
            mappings[old] = new
        logger.info("mappings: " + str(mappings))

    # predefined labels?
    labels = None
    if len(parsed.labels) > 0:
        labels = list(parsed.labels.split(","))
        logger.info("labels: " + str(labels))

    convert(
        input_dir=input_dir, input_files=input_files, output_file=parsed.output, regexp=parsed.regexp,
        mappings=mappings, labels=labels, polygon=parsed.polygon, pretty_print=parsed.pretty_print,
        verbose=parsed.verbose)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as ex:
        print(traceback.format_exc())
