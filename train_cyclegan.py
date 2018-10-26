import os
from arguments import Arguments
from data import CreateDataLoader, CreateRealLoader, CreateSyntheticLoader
from util import save_images
from models.train_model import CycleGanModel
import torch


if __name__ == '__main__':
    args = Arguments().parse()

    # data_loader = CreateDataLoader(args)
    # dataset = data_loader.load_data()

    real_dataset_loader = CreateRealLoader(args)
    synthetic_dataset_loader = CreateSyntheticLoader(args)

    real_dataset = real_dataset_loader.load_data()
    synthetic_dataset = synthetic_dataset_loader.load_data()
    print("Real & Synthetic dataset load completed")

    model = CycleGanModel()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
    model.module.initialize(args)
    print("The model has now been created")

    dataset = list(zip(real_dataset, synthetic_dataset))

    for epoch in range(0, args.n_epochs):
        for i, (real, synthetic) in enumerate(dataset):
            if i == (len(dataset)-1):
                continue
            model.module.set_input(real, synthetic)
            print('Epoch {} : [{} / {}]'.format(epoch, i, len(dataset) - 1))
            model.module.train()
            # visuals = model.get_current_visuals()
            # img_real_path, img_syn_path = model.get_image_paths()
            # img_real_size, img_syn_size = model.get_image_sizes()
            # print('%04d: processing image... %s, %s' % (i, img_real_path, img_syn_path))
            # save_images(args.results_dir, visuals, img_path, size=img_size)
        model.module.save_checkpoint()
