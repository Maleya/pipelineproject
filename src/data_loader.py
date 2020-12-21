# Project course SD2711 Small Craft Design, KTH 2020.
# Examiner: Jakob Kuttenkeuler
# Supervisor: Aldo Teran Espinoza

import os
import zipfile
import cv2
import numpy as np
import argparse


class data_loader():
    def __init__(self, args):
        self.paath2root = os.getcwd()
        self.path2img = "{}/{}".format(os.getcwd(), "img")
        self.path2videos = "{}/{}".format(os.getcwd(), "videos")
        self.args = args
        self.use_fps_video = False
        self.write_images = False

    def extract_data(self):
        """
        Extracting data
        """
        if not os.path.isdir("videos"):
            with zipfile.ZipFile("videos.zip") as zipOBJ:
                zipOBJ.extractall(self.paath2root)

    def video2frames(self):
        """
        Creates frames of videos
        Return:
            - image_list (list): list of images from video
        """
        os.chdir(self.path2videos)
        videos = [(video, video[:-4]) for video in os.listdir(
            self.path2videos) if video.endswith(".mp4")]

        for info in videos:

            video = info[0]
            dirname_video = info[1]

            if not dirname_video == self.args.videoName:
                continue
            if not os.path.isdir(dirname_video):
                os.mkdir(dirname_video)
            videocap = cv2.VideoCapture(video)
            frame_count, fps = videocap.get(
                cv2.CAP_PROP_FRAME_COUNT), videocap.get(cv2.CAP_PROP_FPS)
            duration = int(frame_count/fps)
            sec = 0
            count = 1
            success = self.getFrame(videocap, sec, count, dirname_video)
            image_list = []
            if self.use_fps_video:
                s = 1 / fps
            else:
                s = 1 / self.args.video_fps
            while success and sec <= duration:
                count += 1
                sec = round(sec + s, 2)
                success, image = self.getFrame(
                    videocap, sec, count, dirname_video)
                if image is not None:
                    image_list.append(image)

        return image_list

    def getFrame(self, videocap, sec, count, dirNameVideo):
        """
        get frames of the video, then saved as .jpg
        Input: 
            - videocap
            - sec
            - count (int): used to set frame id
            - dirNameVideo (str): ex sonar_1
        Return:
            - hasFrames (bool)
            - image: frame from video
        """
        videocap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, image = videocap.read()

        h, w, c = np.shape(image)
        margin_h = h // 10
        margin_w = w // 5
        image = image[margin_h: h - margin_h, margin_w: w - margin_w]

        if hasFrames and self.write_images:
            img = "image_000000"
            n = len(str(count))
            m = len(img)
            cv2.imwrite("{}/{}/{}".format(self.path2videos, dirNameVideo,
                                          img[:m-n] + str(count) + ".jpg"), image)
        return hasFrames, image

    def frames2video(self):
        """
        Making videos of frames.
        """

        videos = [(video, video[:-4]) for video in os.listdir(
            self.path2videos) if video.endswith(".mp4")]

        for info in videos:
            if not info[1] == self.args.videoName:
                continue
            img_array = []
            path = "{}/{}".format(self.path2videos, info[1])
            os.chdir(path)
            files = [no.strip("image.jpg") for no in os.listdir(path)]
            files.sort(key=int)
            for filename in files:
                img = cv2.imread(
                    "{}/{}/{}{}{}".format(self.path2videos, info[1], "image", filename, ".jpg"))
                h, w, layers = np.shape(img)[0], np.shape(img)[
                    1], np.shape(img)[2]
                size = (w, h)
                img_array.append(img)

            out = cv2.VideoWriter("{}/{}".format(
                self.path2videos, info[1] + "transform.avi"), cv2.VideoWriter_fourcc(*'XVID'), int(1/self.args.video_fps), size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()


def parser():
    """
    The parser us used to choose video and frame per second
    Default: 
        Video:  sonar_1
        fps:    0.5
    """
    parser = argparse.ArgumentParser(
        description="Settings for preprocessing and selected video")
    parser.add_argument("-v", "--videoName", type=str, default="sonar_1",
                        help="Name of video DIR")
    parser.add_argument("-f", "--video_fps", type=float, default=2,
                        help="Frame per second")

    return parser.parse_args()


if __name__ == "__main__":

    """ 
    * Choose another video or fps by:

        DL.args.videoName = "sonar_3"
        DL.args.video_fps = 1

    * If writing video from frames:
        DL.frames2video()
    * If saving images in dir:
        DL.write_images = True
    * If using the real fps from the video:
        DL.use_fps_video = True

    """
    import matplotlib.pyplot as plt
    args = parser()
    DL = data_loader(args)
    # DL.use_fps_video = True
    DL.extract_data()
    image_list = DL.video2frames()
    print(len(image_list))
    # DL.frames2video()

    # for img in image_list:
    #     plt.imshow(img)
    #     plt.show()
