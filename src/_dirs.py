import os

dir = os.path.dirname(os.path.realpath(__file__))

CURRENT_DIR = dir
PROJECT_DIR = "%s/.." % dir
SRC_DIR = "%s/src" % PROJECT_DIR
DATA_DIR = "%s/data" % PROJECT_DIR
DIST_DIR = "%s/dist" % PROJECT_DIR
TMP_DIR = "%s/tmp" % PROJECT_DIR
