from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2


class VideoWriter:
    def __init__(self, buf_size=120):
        self.bufSize = buf_size
        self.buffer = deque(maxlen=buf_size)
        self.result_video_frames = []
        self.writer = None

    def update_buffer(self, frame):
        '''
        Save buffers frames (10 seconds)
        '''

        self.buffer.append(frame)

    def save_video(self, output_path, fourcc, fps, detect_video_frames):
        '''
        Concatenate buffers frames and video frames
        '''

        self.result_video_frames = list(self.buffer) + detect_video_frames
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (self.result_video_frames[0].shape[1],
                                                                 self.result_video_frames[0].shape[0]), True)
        self.buffer.clear()
        self.write_video()

    def write_video(self):
        '''
        Write video frames to .ivi video format when detected.
        '''

        # keep looping
        if len(self.result_video_frames) != 0:
            print(f"Количество кадров в записи: {len(self.result_video_frames)}")
            print(f"Количество секунд в записи: {len(self.result_video_frames) / 24}")

            for frame in self.result_video_frames:
                self.writer.write(frame)

            self.writer.release()
            self.result_video_frames.clear()
