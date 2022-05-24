from datetime import datetime

import cv2
import imutils

from video_writer import VideoWriter


def get_video_with_motion(if_rtsp = False,
                          camera_rtsp="rtsp://192.168.1.10/user=admin_password=12345678A_channel=1_stream=0.sdp?real_stream"):
    """
    This function write and save video with motion.

    :param camera_rtsp: rtsp-url of camera
    :return: saved video with motion
    """
    contour_min_area = 5000     # minimum area size objects contour
    output = './'   # path to output directory
    fps: int = 20    # FPS of output video
    buffer_size: int = 120  # buffer size of video clip writer

    if if_rtsp:
        vs = cv2.VideoCapture(camera_rtsp)
    else:
        vs = cv2.VideoCapture(0)

    first_frame = None
    avg = None
    video_writer = VideoWriter(buf_size=buffer_size)
    not_occupy_frame_count = 0  # количество кадров без окупации
    max_not_occupy_frame_count = 120    # максимальный промежуток простоя в видеозаписи
    occupy = False  # флаг оккупации
    occupy_video_frames = []    # оккупированные кадры
    update_buffer_before_write_stream = True    # добавление в буфер

    while True:
        ret, frame = vs.read()

        if frame is None:
            break

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

            if cv2.contourArea(contour) < contour_min_area:
                occupy = False
                update_buffer_before_write_stream = True
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text = "Occupied"
            occupy = True
            break

        if update_buffer_before_write_stream and (not occupy and len(occupy_video_frames) == 0):
            video_writer.update_buffer(frame)   # Update buffer before occupied

        if occupy:
            not_occupy_frame_count = 0
            occupy_video_frames.append(frame)

        if not occupy and len(occupy_video_frames) != 0:
            occupy_video_frames.append(frame)

        if not occupy:
            not_occupy_frame_count += 1

        if not_occupy_frame_count > max_not_occupy_frame_count and len(occupy_video_frames) != 0:
            timestamp = datetime.now()
            p = f"{output}/{timestamp.strftime('%Y%m%d-%H%M%S')}.mp4"
            video_writer.save_video(p, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, occupy_video_frames)
            occupy_video_frames.clear()
            not_occupy_frame_count = 0
            print('Video saved.')

        frame = cv2.putText(frame, f"{text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        cv2.imshow("Security Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_video_with_motion()
