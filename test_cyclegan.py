import os
from arguments import Arguments
from data import CreateDataLoader, CreateRealLoader, CreateSyntheticLoader
from util import save_images
from .models.train_model import CycleGanModel


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
    model.initialize(args)
    print("The model has now been created")

    for i in range(0, synthetic_dataset.__len__()):
        real = real_dataset[i]
        synthetic = synthetic_dataset[i]
        model.set_input(real, synthetic)
        print("Progress [{} / {}]", i, synthetic_dataset.__len__())
        model.train()
        visuals = model.get_current_visuals()
        img_real_path, img_syn_path = model.get_image_paths()
        img_real_size, img_syn_size = model.get_image_sizes()
        print('%04d: processing image... %s, %s' % (i, img_real_path, img_syn_path))
        save_images(args.style_results_dir, visuals, img_path, size=img_size)
