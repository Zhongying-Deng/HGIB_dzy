import numpy as np
import os
import ntpath
import time

class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.loss_names = ['cls']
        if opt.focal:
            self.loss_names.append('focal')
        if 'GIB' in self.name:
            self.loss_names.append('kl')
            # self.loss_names.append('kd')
        if opt.weight_center > 0:
            self.loss_names.append('center')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, acc, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        message += '%s: %.3f ' % ('accurcy', acc)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_val_losses(self, epoch, losses, acc, phase="Train"):
        message = '[%s  epoch: %d] ' % (phase, epoch)
        # print(self.loss_names, losses)
        if self.opt.focal:
            for i in range(len(losses)):
                message += '%s: %.3f ' % (self.loss_names[i], losses[i])
        else:
            message += '%s: %.3f ' % (self.loss_names[0], losses[0][0])
        message += '%s: %.3f ' % ('accurcy', acc)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
