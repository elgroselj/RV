import argparse

from evaluate_trackerLT import evaluate_tracker


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
parser.add_argument("--verbose", help="Verbose", required=False, action='store_true')
parser.add_argument("--treshold_stop", help="Treshold for stop tracking", required=False, action='store')
parser.add_argument("--treshold_start", help="Treshold for start tracking", required=False, action='store')
parser.add_argument("--sigma", help="Dispersion of candidates.", required=False, action='store')
parser.add_argument("--N", help="Number of candidates.", required=False, action='store')
parser.add_argument("--locality", help="Number of tries in local area, before forgetting about it.", required=False, action='store')

args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize, args.verbose, float(args.treshold_stop), float(args.treshold_start), float(args.sigma), int(args.N), int(args.locality))
