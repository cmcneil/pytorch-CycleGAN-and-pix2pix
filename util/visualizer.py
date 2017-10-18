import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

    def display_current_heatmaps(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, np_data in visuals.items():
                #image_numpy = np.flipud(image_numpy)
                self.vis.heatmap(np_data[0, :, :].transpose(),
                                 opts=dict(title=label, colormap='bwr', xmin='-3.0', xmax='3.0'),
                                 win=self.display_id + idx)
                idx += 1

    def display_current_ims(self, ims, epoch):
        if self.display_id > 0:  # show images in the browser
            idx = 2
            for label, image_numpy in ims.items():
                # image_numpy = (image_numpy * 255.0).astype(np.uint8)
                print label
                print 'shape im: ' + str(np.shape(image_numpy))
                print 'max im: ' + str(np.max(image_numpy))
                self.vis.image(image_numpy.transpose([0, 2, 1]), #[::-1, ...],
                               opts=dict(title=label),
                               win=self.display_id + idx)
                idx += 1

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, image_numpy in visuals.items():
                #image_numpy = np.flipud(image_numpy)
                im_disp = image_numpy.transpose([2, 0, 1]).astype(np.float32)/255.0
                print "disp curr results: " + str(np.shape(im_disp)) + ", " + str(np.max(im_disp))
                self.vis.image(im_disp, opts=dict(title=label),
                               win=self.display_id + idx)
                idx += 1

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    def plot_stopping_set_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'ss_plot_data'):
            self.ss_plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.ss_plot_data['X'].append(epoch + counter_ratio)
        self.ss_plot_data['Y'].append([errors[k] for k in self.ss_plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.ss_plot_data['X'])]*len(self.ss_plot_data['legend']),1),
            Y=np.array(self.ss_plot_data['Y']),
            opts={
                'title': 'Stopping set losses',
                'legend': self.ss_plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=6)

    def plot_current_filters(self, epoch, opt, filters):
        nfilt = np.shape(filters)[-1]
        ntime = np.shape(filters)[-2]
        self.vis.line(
            X=np.stack([np.arange(ntime)]*nfilt, 1),
            Y=filters,
            opts={'title': 'compression filters',
                  'xlabel': 'tr#'},
            win=5
        )

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, start_time):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, time.time() - start_time)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
