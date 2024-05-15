import argparse
import os
import cv2
from random import randint
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from siamfc import TrackerSiamFC

def redetection_score(score, candidate, last_detected_center, n_fails):
    dist = np.linalg.norm(candidate - last_detected_center, 2)
    return score*(1/dist)*n_fails
    
    


def evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality):
    
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    tracker = TrackerSiamFC(net_path=network_path)

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization
        mode = "tracking"
        candidate_predictions = []
        if verbose:
            print("INFO: Into tracking mode.")

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):
            img = cv2.imread(sequence.frame(i))
            last_detected = results[-1]
            
            if mode == "tracking":
                candidate_predictions = []
                prediction, score = tracker.update(img)
                if verbose:
                    print("score:",score)
                
                if score < treshold_stop:
                    mode = "redetection"
                    n_fails = 0
                    if verbose:
                        print("INFO: Into redetection mode.")
                    results.append(last_detected)
                    scores.append([score])
                else:
                    results.append(prediction)
                    scores.append([score])
                    
                        
                        
            elif mode == "redetection":
                n_fails += 1
                best = None
                height, width, _ = img.shape
                # candidates = np.array(np.random.multivariate_normal(tracker.center, sigma * np.eye(2), N))
                if n_fails <= locality:
                    candidates = np.array(np.random.multivariate_normal(tracker.center, np.sqrt(n_fails+1)*(height+width)/2*sigma * np.eye(2), N))
                    candidates = np.array([np.clip(candidates[:,0],0,height), np.clip(candidates[:,1],0,width)]).transpose()
                    candidates = list(candidates) 
                else:
                    candidates = [np.array([randint(0,height), randint(0,width)],dtype="float64") for i in range(N)]
                    
                candidate_predictions = []
                for candidate in candidates:
                    prediction, score0, center, scale = tracker.redetect(img,candidate)
                    score = redetection_score(score0,candidate,tracker.center,n_fails)
                    
                    candidate_predictions.append(prediction)
                    if best is None or best[1] < score:
                        best = [prediction, score, center, scale]
                prediction, score, center, scale = best
                if verbose:
                    print("score:",score)
                if score > treshold_start:
                    mode = "tracking"
                    if verbose:
                        print("INFO: Into tracking mode.")
                    tracker.apply_detection(img, center, scale)
                    results.append(prediction)
                    scores.append([score])
                else:
                    results.append(last_detected)
                    scores.append([score])
                    
                    
                    
            if visualize or verbose:
                if mode == "tracking":
                    tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                    br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)

                    cv2.imshow('win', img)
                    key_ = cv2.waitKey(10)
                    if key_ == 27:
                        exit(0)
                        
                elif mode == "redetection":
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # org = (50, 50)
                    # fontScale = 1
                    # color = (255, 0, 0)
                    # thickness = 2
                    # img = cv2.putText(img, 'NA', org, font,  
                    #                 fontScale, color, thickness, cv2.LINE_AA) 
                    for prediction in candidate_predictions:
                        tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                        br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                        cv2.rectangle(img, tl_, br_, (255, 0, 0), 1)
                    
                    cv2.imshow('win', img)
                    key_ = cv2.waitKey(10)
                    if key_ == 27:
                        exit(0)
                    
                
        
        save_results(results, bboxes_path)
        save_results(scores, scores_path)


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
