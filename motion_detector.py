import argparse
import imutils
import time
from datetime import datetime
import cv2

from video_writer import VideoWriter

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=0, help="path to the video file")
ap.add_argument("-a", "--contour-min-area", type=int, default=5000, help="minimum area size objects contour")

ap.add_argument("-o", "--output", required=False, default='./', help="path to output directory")
ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="mp4v", help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=5, help="buffer size of video clip writer")

args = vars(ap.parse_args())

if args["video"]:
    vs = cv2.VideoCapture(args["video"])
else:
    vs = cv2.VideoCapture(0)

#"rtsp://192.168.1.10/user=admin_password=12345678A_channel=1_stream=0.sdp?real_stream"

first_frame = None
avg = None
video_writer = VideoWriter(buf_size=args["buffer_size"])
not_occupy_frame_count = 0  # количество кадров без окупации
max_not_occupy_frame_count = 120    # максимальный промежуток простоя в видеозаписи
occupy = False  # флаг оккупации
occupy_video_frames = []    # оккупированные кадры
update_buffer_before_write_stream = True    # добавление в буфер

while True:
    if args["video"]:
        frame = vs.read()
    else:
        ret, frame = vs.read()

    if frame is None:
        break

    # frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    frame = imutils.resize(frame, width=1500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    if first_frame is None:
        first_frame = gray
        continue

    avg = cv2.accumulateWeighted(gray, avg, 0.15)

    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        update_buffer_before_write_stream = False

        if cv2.contourArea(contour) < args["contour_min_area"]:
            occupy = False
            update_buffer_before_write_stream = True
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = "Occupied"
        occupy = True
        break

    if update_buffer_before_write_stream and (occupy and len(occupy_video_frames) == 0):
        video_writer.update_buffer(frame)   # Update buffer before occupied

    if occupy:
        not_occupy_frame_count = 0
        occupy_video_frames.append(frame)

    if not occupy and len(occupy_video_frames) != 0:
        occupy_video_frames.append(frame)

    if not occupy:
        not_occupy_frame_count += 1

    # if not_occupy_frame_count < max_not_occupy_frame_count:
    #     occupy_video_frames.append(frame)
    if not_occupy_frame_count > max_not_occupy_frame_count and len(occupy_video_frames) != 0:
        timestamp = datetime.now()
        p = f"{args['output']}/{timestamp.strftime('%Y%m%d-%H%M%S')}.avi"
        video_writer.save_video(p, cv2.VideoWriter_fourcc(*args["codec"]), args["fps"], occupy_video_frames)
        occupy_video_frames.clear()
        not_occupy_frame_count = 0
        print('Video saved.')

    frame = cv2.putText(frame, f"{text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
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
