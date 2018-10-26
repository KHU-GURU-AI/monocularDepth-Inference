import os
from arguments import Arguments
from data import CreateDataLoader, CreateRealLoader, CreateSyntheticLoader
from util import save_images
from models.train_model2 import GanModel
import torch


if __name__ == '__main__':
    args = Arguments().parse()

    # data_loader = CreateDataLoader(args)
    # dataset = data_loader.load_data()

    synthetic_dataset_loader = CreateSyntheticLoader(args)
    synthetic_dataset = synthetic_dataset_loader.load_data()
    print("Synthetic & Depth dataset load completed")

    model = GanModel()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.module.initialize(args)
    print("The model has now been created")

    for epoch in range(0, args.n_epochs):
        for i, data in enumerate(synthetic_dataset):
            model.module.set_input(data)
            print('Epoch {} : [{} / {}]'.format(epoch, i, synthetic_dataset.__len__() / args.batchSize))
            model.module.train()
            # visuals = model.get_current_visuals()
            # img_real_path, img_syn_path = model.get_image_paths()
            # img_real_size, img_syn_size = model.get_image_sizes()
            # print('%04d: processing image... %s, %s' % (i, img_real_path, img_syn_path))
            # save_images(args.results_dir, visuals, img_path, size=img_size)
            # if i == 1:
            #     break
        model.module.save_checkpoint()
