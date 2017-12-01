from argparse import Namespace
import time
import sys
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

# Prepare the stopping set, to look at test set progress.
modopt = Namespace(**vars(opt))
vars(modopt)['phase'] = 'test'
print modopt.phase
stopping_set_loader = CreateDataLoader(modopt)
stopping_set = iter(stopping_set_loader.load_data())

total_steps = 0
stop_idx = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    try:
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            if i == 0 and epoch == 1:
                model.optimize_parameters()
            # else:
            model.optimize_parameters(only_d=False)

            if total_steps % opt.display_freq == 0:
                # print model.get_current_visuals()
                visualizer.display_current_heatmaps(model.get_current_visuals(), epoch)
                visualizer.display_current_ims(model.get_current_ims(), epoch)
                # visualizer.plot_current_filters(epoch, opt, model.get_filters())

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size,
                                                   opt, errors)
                    if i % 12 == 0:
                        stopping_set = iter(stopping_set_loader.load_data())
                    val_dset = stopping_set.next()
                    stopping_set_errors = model.get_errors_for_input(val_dset)

                    visualizer.plot_stopping_set_errors(epoch, float(epoch_iter)/dataset_size,
                                                        opt, stopping_set_errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    except KeyboardInterrupt:
        model.save('interrupted')
        sys.exit(0)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
