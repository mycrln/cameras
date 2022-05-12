import argparse
import imutils
import time
from datetime import datetime
import cv2

from video_writer import KeyClipWriter

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=0, help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="minimum area size")

ap.add_argument("-o", "--output", required=False, default='./', help="path to output directory")
ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="mp4v", help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=5, help="buffer size of video clip writer")

args = vars(ap.parse_args())

if args["video"]:
    vs = cv2.VideoCapture(args["video"])
else:
    vs = cv2.VideoCapture("rtsp://192.168.1.10/user=admin_password=12345678A_channel=1_stream=0.sdp?real_stream")

first_frame = None
avg = None
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consec_frames = 0
not_ocup_count = 0
not_ocup_max = 1000
occupy = False
ocup_video_frames = []

while True:
    if args["video"]:
        frame = vs.read()
    else:
        ret, frame = vs.read()

    # frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    update_consec_frames = True
    if update_consec_frames:
        kcw.update(frame)   # Update buffer before occupied

    if frame is None:
        break

    frame = imutils.resize(frame, width=1500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    if first_frame is None:
        first_frame = gray
        continue

    cv2.accumulateWeighted(gray, avg, 0.15)

    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=5)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if occupy is False:
        not_ocup_count += 1

    for c in cnts:

        if cv2.contourArea(c) < args["min_area"]:
            update_consec_frames = False
            occupy = False
            continue

        occupy = True
        update_consec_frames = True
        ocup_video_frames.append(frame)

        # Delete frames if not ocupied
        if not_ocup_count > not_ocup_max:
            not_ocup_count = 0
            if kcw.recording:
                kcw.finish()
            ocup_video_frames.clear()

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = "Occupied"

        # print(datetime.now())
        # update the key frame clip buffer
        # if we are recording and reached a threshold on consecutive
        # number of frames with no action, stop recording the clip
        #if kcw.recording:# and consecFrames == args["buffer_size"]:

    if not kcw.recording:
        timestamp = datetime.now()
        p = f"{args['output']}/{timestamp.strftime('%Y%m%d-%H%M%S')}.avi"
        kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]), args["fps"], ocup_video_frames)

    cv2.putText(frame, f"Room Status: {text}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Security Feed", frame)
    #time.sleep(0.2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if args["video"]:
    vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()