"""
Copyright 2018 University of Waikato, Hamilton, NZ

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

import argparse
import os
import traceback
import logging
import PIL.Image as pil
from adams.report import read_objects, determine_labels, fix_labels
from adams.report import SUFFIX_TYPE, SUFFIX_X, SUFFIX_Y, SUFFIX_WIDTH, SUFFIX_HEIGHT, REPORT_EXT
from adams.report import PREFIX_OBJECT, DEFAULT_LABEL

# logging setup
logging.basicConfig()
logger = logging.getLogger("adams.convert")
logger.setLevel(logging.INFO)


def convert(input_dir, input_files, output_file, mappings=None, regexp=None, labels=None, verbose=False):
    """
    Converts the images and annotations (.report) files into TFRecords.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param input_files: the file containing the report files to use
    :type input_files: str
    :param output_file: the output file for TFRecords
    :type output_file: str
    :param mappings: the label mappings for replacing labels (key: old label, value: new label)
    :type mappings: dict
    :param regexp: the regular expression to use for limiting the labels stored
    :type regexp: str
    :param labels: the predefined list of labels to use
    :type labels: list
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    """
    pass


def main():
    """
    Runs the conversion from command-line. Use -h/--help to see all options.
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
        "-v", "--verbose", action="store_true", dest="verbose", required=False,
        help="whether to be more verbose when generating the records")
    parsed = parser.parse_args()

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

    # predefined labels?
    labels = None
    if len(parsed.labels) > 0:
        labels = list(parsed.labels.split(","))
        logger.info("labels: " + str(labels))

    if parsed.verbose:
        logger.info("sharding off" if parsed.shards <= 1 else "# shards: " + str(parsed.shards))

    convert(
        input_dir=input_dir, input_files=input_files, output_file=parsed.output, regexp=parsed.regexp,
        mappings=mappings, labels=labels, verbose=parsed.verbose)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(traceback.format_exc())
