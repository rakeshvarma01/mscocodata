# Flows for generating multiple Detectron compatible datasets and training/testing on them

# Detectron Data Generator Flow

Before running, all images need to be annotated and all report files available

The user is prompted to enter the following:
1- Image Directory: Path to the directory containing the images and report files
2- Output Directory: Path to the directory to store the resulted images and json annotations files
3- Number of Folds: Number of folds to produce, less than 2 will perform a leave one out cross validation
4- Number of Rows: Number of rows to cut the images to
5- Number of Columns: Number of columns to cut the images to
6- Overlap in Pixels: Number of pixels to overlap between cut images
7- Python Executable: Path to the Python executable file used to execute the Python script
8- MSCOCO Script Directory: Top level directory of the "convert ADAMS reports to json" Python script

After finishing, the output directory will contain multiple directories, each with the following directories:
1- Annotations: Contains the json annotations file for training images
2- Train: Contains the cut training images
3- Val: Contains the cut test image(s) and json annotations file for them


# Detectron Auto Train Test Flow

Before using this flow for the first time, do the following:
1- Make sure that Detectron is correctly installed and working
2- Open "Path_to_Detectron/detectron/datasets/dataset_catalog.py" and add
,
    'generic_train': {
        _IM_DIR: 'Path/to/train/images',  # generic_train_dir
        _ANN_FN: 'Path/to/train/annotations'  # generic_train_annotations
    },
    'generic_val': {
        _IM_DIR: 'Path/to/test/images',  # generic_val_dir
        _ANN_FN: 'Path/to/test/annotations'  # generic_val_annotations
    }
to the end of _DATASETS
The first , is added after } or an error would occur
3- Copy any yaml file from "Path_to_Detectron/configs" directory, rename it to suit your project, and then make the following changes:
a- Modify NUM_CLASSES to "classes + 1 (background)"
b- Add "VIS: True" after NUM_GPUS
c- Download a pretrained model suiting your alogorithm from Detectron website, and add its path to WEIGHTS
d- Change DATASETS in TRAIN to ('generic_train',) and in TEST to ('generic_val',)
e- Add "FORCE_JSON_DATASET_EVAL: True" to the end of TEST
f- Add "OUTPUT_DIR: Path/to/output/directory # Output Directory" to the end of file

The user is prompted to enter the following:
1- Datasets Directory: Top level directory containing all the datasets
2- Output Directory: Path to the directory to store trained models and test results
3- Config Yaml File: Path to the yaml config file
4- Python Executable: Path to the Python executable file used to execute the Python script
5- Detectron Script Directory: Top level directory of the "tools/train_net.py" Python script

After finishing, the output directory will contain multiple directories, each with the following directories:
1- Test: Contains the test result as a pdf file, can be found in vis directory
2- Train: Contains the trained models