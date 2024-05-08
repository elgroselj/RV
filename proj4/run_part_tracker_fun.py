import time
import cv2

from sequence_utils import VOTSequence
from part_tracker import PartParams,PartTracker

def run(sequence="bolt",parameters=PartParams(),plot=True,video_delay=15,verbose=True):
    # set the path to directory where you have the sequences
    dataset_path = '/home/lema/Documents/RV/proj3/vot2013/' # TODO: set to the dataet path on your disk
    sequence = sequence  # choose the sequence you want to test

    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    show_gt = False
    # video_delay = 15

    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    overlap = [0,0]
    init_times = []
    track_times = []
    # create parameters and tracker objects
    
    tracker = PartTracker(parameters)
    


    time_all = 0

    # initialize visualization window
    if plot:
        sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            t_diff = time.time() - t_
            init_times.append(t_diff)
            time_all += t_diff
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            t_diff = time.time() - t_
            track_times.append(t_diff)
            time_all += t_diff

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)
        overlap[0]+=o
        overlap[1]+=1

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if plot:
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
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
    if verbose:
        print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
        print('Tracker failed %d times' % n_failures)
        print('Average overlap %d percent' % (overlap[0]/overlap[1]*100))
    return((sequence.length() / time_all), n_failures, overlap[0]/overlap[1], init_times, track_times)

run()
