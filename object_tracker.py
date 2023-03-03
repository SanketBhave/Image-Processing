import cv2
import sys

print(sys.version)

video1 = cv2.VideoCapture('pa01/Dusty_snow.mp4')
video2 = cv2.VideoCapture('pa01/frigatebird.mp4')
video3 = cv2.VideoCapture('pa01/Merlin_run.mp4')
videos = [video1, video2, video3]


def track(img_tracker):
    for video in videos:
        ok, frame = video.read()
        cv2.namedWindow("First frame", cv2.WINDOW_NORMAL)
        box = cv2.selectROI("First frame", frame)
        init_tracker = tracker.init(frame, box)
        while True:
            ok, frame = video.read()
            if ok:
                ret, box = img_tracker.update(frame)
                point1 = (int(box[0]), int(box[1]))
                point2 = (int(box[0]+box[2]), int(box[1]+box[3]))
                cv2.rectangle(frame, point1, point2, (0, 0, 255), 2)
                cv2.namedWindow("Object Detected", cv2.WINDOW_NORMAL)
                cv2.imshow("Object Detected", frame)
                k = cv2.waitKey(1) & 0xff
                if k == 27: break


if __name__ == "__main__":
    choice = int(input("Select a tracking option" + '\n' + "1] MIL tracker" + '\n' + "2] Goturn tracker" + '\n' + "3] Dasiam tracker"))
    if choice == 1:
        tracker = cv2.TrackerMIL_create()
    elif choice == 2:
        tracker = cv2.TrackerGOTURN_create()
    elif choice == 3:
        tracker = cv2.legacy.TrackerKCF_create()
    elif choice == 4:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif choice == 5:
        tracker = cv2.legacy.TrackerBoosting_create()
    track(tracker)
