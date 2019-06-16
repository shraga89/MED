import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import config as conf


# from keras import backend as K


def grayscale_to_rgb(img):
    return np.stack((img,) * 3, axis=-1)


class mouseHandler:
    max_x = 1279.0
    max_y = 1023.0

    def __init__(self, matcherId, M=None):
        self.matcherId = matcherId
        if not M:
            self.mouseData = {}
            self.newMouseData = []
            self.dir = conf.dir
            self.processMouseData()
            self.list2dict()
        else:
            self.mouseData = {}
            self.newMouseData = M
            self.list2dict()

    def processMouseData(self):
        last_line = None
        try:
            with open(str(self.dir + 'ExperimentData/' + self.matcherId + '/Excel - CIDX/2.rms')) as f:
                for line in f.readlines():
                    line.replace('{', '').replace('}', '').split()
                    action, value = line.replace('{', '').replace('}', '').split()[:2]
                    if last_line:
                        delay = last_line.replace('}', '').split()[1]
                        if not '.' in delay:
                            delay = 0.0
                    else:
                        delay = ''
                    # if action not in self.mouseData:
                    #     self.mouseData[action] = list()
                    if action == 'Move':
                        p1, p2 = value.replace('(', '').replace(')', '').split(',')[:2]
                        # self.mouseData[action] += [tuple((tuple((float(p1), float(p2))), delay)), ]
                        self.newMouseData += [tuple((action, tuple((float(p1), float(p2))), delay)), ]
                    elif 'Mouse' in action:
                        add = line.replace('{', '').replace('}', '').split()[1:3]
                        if 'down' in add:
                            p1, p2 = add[1].replace('(', '').replace(')', '').split(',')[:2]
                            # self.mouseData[action] += [tuple((tuple((float(p1), float(p2))), delay)), ]
                            self.newMouseData += [tuple((action, tuple((float(p1), float(p2))), delay)), ]
                    elif 'Delay' in action:
                        # self.mouseData[action] += [value, ]
                        if isinstance(delay, str):
                            delay = 0.0
                        self.newMouseData += [tuple((action, None, delay)), ]
                    else:
                        # self.mouseData[action] += [value, ]
                        self.newMouseData += [tuple((action, value, delay)), ]
                    last_line = line
        except:
            print('couldnt find data for' + self.matcherId)

    def list2dict(self):
        for a in self.newMouseData:
            action = a[0]
            if action not in self.mouseData:
                self.mouseData[action] = list()
            self.mouseData[action] += [tuple(a[1:]), ]

    def exportMouseData(self, method, save_heatmaps=False):
        # USE THE LIST!!
        x = []
        y = []
        weights = []
        if method not in self.mouseData:
            return np.zeros((37, 45, 3))
        for k in self.mouseData[method]:
            try:
                d = float(k[1])
            except:
                continue
            i, j = k[0]
            x += [float(i), ]
            y += [float(j), ]
            weights += [d, ]
        if len(x) == 0 or len(y) == 0:
            return
        xedges = list(range(0, int(mouseHandler.max_x) + 100, 30))
        yedges = list(range(0, int(mouseHandler.max_y) + 100, 30))
        # heatmap, _, _ = np.histogram2d(x, y, bins=(xedges, yedges), weights=weights)
        heatmap, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap = heatmap.T
        if save_heatmaps:
            plt.clf()
            map_img = plt.imread(self.dir + 'screen.jpg')
            hmax = sns.heatmap(heatmap,
                               cmap='Reds',
                               alpha=0.5,  # whole heatmap is translucent
                               zorder=2,
                               cbar=False
                               )
            hmax.imshow(map_img,
                        aspect=hmax.get_aspect(),
                        extent=hmax.get_xlim() + hmax.get_ylim(),
                        zorder=1)  # put the map under the heatmap
            plt.axis('off')
            if not os.path.exists('./figs/' + method):
                os.makedirs('./figs/' + method)
            plt.savefig('./figs/' + method + '/' + self.matcherId + '.jpg', bbox_inches='tight', format='jpg', dpi=300)
        return grayscale_to_rgb(heatmap)

    def split2ns(self, matchers):
        M_list = self.newMouseData
        M_dict = self.mouseData
        sub_matchers_size = len(matchers)
        bucket_size = int(len(self.newMouseData) / sub_matchers_size)
        submouses = {}
        last = 0
        for i, m in enumerate(matchers):
            submouse = M_list[last: (i + 1) * bucket_size]
            last = (i + 1) * bucket_size
            submouses[m] = mouseHandler(m, submouse)
        return submouses

    def extract_mouse_features(self):
        total_length = float(len(self.newMouseData))
        total_actions = float(len(self.mouseData.keys()))
        min_x, min_y, max_x, max_y, sum_x, count_pos, sum_y, \
        total_time, total_dist, max_speed = [0.0, ] * 10
        i = 0
        while i < len(self.newMouseData):
            currElapsedTime = 0.0
            if self.newMouseData[i][0] == 'Delay':
                currElapsedTime += float(self.newMouseData[i][2])
                i += 1
            if i >= len(self.newMouseData): break
            while not isinstance(self.newMouseData[i][1], tuple):
                currElapsedTime += float(self.newMouseData[i][2])
                i += 1
            if i >= len(self.newMouseData): break
            currElapsedTime += float(self.newMouseData[i][2])
            j = i + 1
            if j >= len(self.newMouseData): break
            if self.newMouseData[j][0] == 'Delay':
                j += 1
            if j >= len(self.newMouseData): break
            while not isinstance(self.newMouseData[j][1], tuple):
                currElapsedTime += float(self.newMouseData[j][2])
                j += 1
                if j >= len(self.newMouseData): break
            if j >= len(self.newMouseData): break
            # print(self.newMouseData[i], self.newMouseData[j])
            currElapsedTime += float(self.newMouseData[j][2])
            currDist = dist(self.newMouseData[i][1], self.newMouseData[j][1])
            currElapsedTime = currElapsedTime * 60
            total_time += currElapsedTime
            total_dist += dist(self.newMouseData[i][1], self.newMouseData[j][1])
            if currElapsedTime > 0.0:
                if currDist / currElapsedTime > max_speed:
                    max_speed = currDist / currElapsedTime
            x_i, y_i = self.newMouseData[i][1]
            sum_x += x_i
            sum_y += y_i
            count_pos += 1
            if x_i < min_x:
                min_x = x_i
            if x_i > max_x:
                max_x = x_i
            if y_i < min_y:
                min_y = y_i
            if y_i > max_y:
                max_y = y_i
            i = j
        avg_speed = 0.0
        if total_time > 0.0:
            avg_speed = total_dist / total_time
        avg_x = 0.0
        avg_x = sum_x / count_pos
        avg_y = sum_y / count_pos
        return total_length, total_actions, total_time, total_dist, \
               max_speed, min_x, min_y, max_x, max_y, avg_speed, avg_x, avg_y


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
