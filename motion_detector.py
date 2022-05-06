import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="minimum area size")
args = vars(ap.parse_args())


if args["video"]:
    vs = cv2.VideoCapture(args["video"])
else:
    vs = cv2.VideoCapture("rtsp://192.168.1.10/user=admin_password=12345678A_channel=1_stream=0.sdp?real_stream")


firstFrame = None
avg = None

while True:
    if args["video"]:
        frame = vs.read()
    else:
        ret, frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    if frame is None:
        break

    frame = imutils.resize(frame, width=1500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    if firstFrame is None:
        firstFrame = gray
        continue

    cv2.accumulateWeighted(gray, avg, 0.15)

    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=5)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:

        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = "Occupied"

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
