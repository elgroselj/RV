import time

import cv2

from sequence_utils import VOTSequence
# from ncc_tracker_example import NCCTracker, NCCParams

import sys
sys.path.append(".")
from ms_tracker import MeanShiftTracker, MSParams

# def reshape(image,region,k=None):
#     col, row, delta_col, delta_row = region
#     if k is None:
#         k0 = 50/delta_row
#         k1 = 50/delta_col
#         k = (k0,k1)
#     region_ = (col*k[1], row*k[0], delta_col*k[1], delta_row*k[0])
#     image_ = cv2.resize(image, (0,0), fx = k[1], fy = k[0])
#     return image_, region_, k

def reshape(image,region,k=None):
    return image, region, (1,1)

def run_ms_tracker_fun(sequence,d={},plot=False,video_delay=15,show_gt = False):
    dataset_path = '/home/lema/Documents/RV/proj2'
    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    font = cv2.FONT_HERSHEY_PLAIN

    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    # parameters = NCCParams()
    # tracker = NCCTracker(parameters)
    parameters = MSParams()
    tracker = MeanShiftTracker(parameters)
    
    tracker.set_params(d)
    print(tracker.parameters.alpha)

    time_all = 0

    # initialize visualization window
    if plot:
        sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    k = None
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        region = sequence.get_annotation(frame_idx, type='rectangle')
        img2, region2, k = reshape(img,region,k=k)
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img2, region2)
            time_all += time.time() - t_
            predicted_bbox2 = region2
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox2 = tracker.track(img2)
            time_all += time.time() - t_

        
        _, predicted_bbox, _ = reshape(img,predicted_bbox2,k=(1/k[0],1/k[1]))
        
        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = region
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        if plot:
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
            sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1
            k = None

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)

    return n_failures

# set the path to directory where you have the sequences
# sequence = 'bolt1'  # choose the sequence you want to test
# sequence = 'basketball'  # choose the sequence you want to test
# sequence = 'bicycle'  # choose the sequence you want to test
# sequence = 'drunk'  # choose the sequence you want to test
# sequence = 'sphere'  # choose the sequence you want to test
# sequence = 'surfing'  # choose the sequence you want to test



# run_ms_tracker_fun(sequence)


