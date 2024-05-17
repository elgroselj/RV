import argparse
import os
import math
import cv2
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import read_results
import matplotlib.pyplot as plt


from performance_evaluation_help import evaluate_performance


parser = argparse.ArgumentParser(description='Longe-term Tracking Performance Evaluation Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')

args = parser.parse_args()

evaluate_performance(args.dataset, args.results_dir)
