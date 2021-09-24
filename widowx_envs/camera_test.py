import cv2
import os
import queue
import threading
import time
import rospy
import sys


class VideoCapture:
    streams = {}

    def __init__(self, name):
        print("starting video stream", name)
        self.name = name
        if name in self.streams:
            print("already started, returning stream")
            return
        cap = cv2.VideoCapture(name)
        if not cap.isOpened():
            print("Error opening resource: " + str(name))
            print("Maybe opencv VideoCapture can't open it")
            exit(0)
        q = queue.Queue()
        self.streams[name] = (cap, q)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()  # read frames as soon as they are available, keeping only most recent one

    def _reader(self):
        while True:
            cap, q = self.streams[self.name]
            ret, frame = cap.read()
            if not ret:
                break
            if not q.empty():
                try:
                    q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            q.put(frame)

    def read(self):
        cap, q = self.streams[self.name]
        return q.get()


if __name__ == '__main__':
    rospy.init_node("camera_delay_test")
    caps = [VideoCapture(1), VideoCapture(3), VideoCapture(5)]
    start = time.time()
    now_trigger = 1

    img_paths = []
    for i in range(len(caps)):
        img_paths.append(os.path.expanduser(f'~/images{i}'))
        os.makedirs(img_paths[i], exist_ok=True)

    while not rospy.is_shutdown():
        time_diff = time.time() - start
        print(time_diff)
        time.sleep(0.05)
        if time_diff > now_trigger:
            imgs = []
            for cap in caps:
                img = cap.read()
                imgs.append(img)
            for img, img_path in zip(imgs, img_paths):
                cv2.imwrite(f'{img_path}/img{now_trigger}.jpg', img)
            now_trigger += 1