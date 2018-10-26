import unittest
import os
import shutil
from PIL import Image
from pprint import pprint as pp

import single_dataset

class TestKITTIDataset(unittest.TestCase):

    def setUp(self):
        # make kitti mock dirs
        self.root_dir = 'test_dir'
        self.origin_dir = 'data_original'

        self.label = []

        for path in ['2011_09_26', '2011_09_28']:
            for sub in ['_drive_0001_sync', '_drive_0002_sync']:
                for sub_sub in ['image_00', 'image_01', 'image_02', 'image_03']:
                    leaf_path = os.path.join(self.root_dir, self.origin_dir, path, sub, sub_sub, 'data')
                    os.makedirs(leaf_path)

                    for img_name in ['0000000000.png', '0000000001.png']:

                        full_img_name = os.path.join(leaf_path, img_name)

                        if sub_sub in ['image_02', 'image_03']:
                            img = Image.new('RGB', (5, 5))
                            img.save(full_img_name)
                            if sub_sub == 'image_02':
                                self.label.append(full_img_name)

                        elif sub_sub in ['image_00', 'image_01']:
                            img = Image.new('L', (5, 5))
                            img.save(full_img_name)

    def tearDown(self):
        shutil.rmtree(self.root_dir)

    def test_make_kitti_dataset(self):
        # test
        dataset = single_dataset.KITTIDataset()
        imgs = dataset.make_kitti_dataset(self.root_dir)

        assert set(self.label) == set(imgs)

if __name__ == '__main__':
    unittest.main()