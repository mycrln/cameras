import imutils
from datetime import datetime
import cv2

from video_writer import VideoWriter

vs = cv2.VideoCapture("rtsp://192.168.1.10/user=admin_password=12345678A_channel=1_stream=0.sdp?real_stream")

first_frame = None
video_writer = VideoWriter(buf_size=240)
update_buffer = True    # добавление в буфер
frames = []

while True:
    ret, frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=1500)
    frames.append(frame)
    if update_buffer:
        video_writer.update_buffer(frame)   # Update buffer before occupied

    if len(frames) == 2*60*24:
        timestamp = datetime.now()
        p = f"./{timestamp.strftime('%Y%m%d-%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer.save_video(p, fourcc, 24, [])
        frames.clear()
        print('Video saved.')

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
