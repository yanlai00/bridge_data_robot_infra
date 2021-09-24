import matplotlib.pyplot as plt
import sys
import numpy as np
import _pickle as pickle


class Getdesig(object):
    def __init__(self, img):
        plt.switch_backend('Qt5Agg')
        self.im_shape = img.shape[:2]

        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)
        plt.imshow(img)

        cid = fig.canvas.mpl_connect('button_press_event', self.    onclick)
        self.i_click = 0

        # self.marker_list = ['o',"D","v","^"]

        self.i_click_max = 1
        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        import matplotlib.pyplot as plt
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)

        print('iclick', self.i_click)

        if self.i_click == self.i_click_max:
            plt.close('all')
            return

        rc_coord = np.array([event.ydata, event.xdata])

        self.desig = rc_coord
        color = "r"
        marker = 'o'
        self.ax.scatter(rc_coord[1], rc_coord[0], s=100, marker=marker, facecolors=color)

        plt.draw()

        self.i_click += 1