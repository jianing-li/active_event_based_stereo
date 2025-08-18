import numpy as np


def make_color_histo(events, img=None, width=640, height=480):
    """
    simple display function that shows negative events as green dots and positive as red one
    on a white background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 255
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        ON_index = np.where(events['p']==1)
        img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220]* events['p'][ON_index][:, None] # red [0, 0, 255]

        OFF_index = np.where(events['p']==0)
        img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30]* (events['p'][OFF_index]+1)[:, None] # green [0, 255, 0], blue [255, 0, 0]

    return img