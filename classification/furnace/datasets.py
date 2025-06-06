import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from furnace.transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from furnace.masking_generator import MaskingGenerator, RandomMaskingGenerator
from furnace.dataset_folder import ImageFolder
import pandas as pd

class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transforms=None):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age

		"""
        self.df = df
        self.transforms = transforms
        self.train = train

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        im_path = filename
        # Use PIL to load the image directly in RGB format
        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        # Apply transformations if any
        if x is not None and self.transforms:
            x = self.transforms(x)

            # x=x.to(torch.float64)
        # Handle whether it's training mode or not
        if self.train:
            y = self.df.iloc[index]['label']
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)

class DataAugmentationForCAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.color_jitter > 0:
            self.common_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        a=args.discrete_vae_type
        print('??????',a)
        if (args.discrete_vae_type == 'clip' or args.discrete_vae_type == 'biomedclip_base16' or \
              args.discrete_vae_type == 'eva-clip-l14' or args.discrete_vae_type == 'monet'):
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()
        
        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )
        

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)

        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_cae_pretraining_dataset(args):
    transform = DataAugmentationForCAE(args)


    molemap_df = pd.read_csv(args.csv_path)
    half_rows = int(len(molemap_df) * args.percent_data)
    molemap_df = molemap_df.head(half_rows)
    dataset_train = SkinDataset(df=molemap_df,
                                train=True,
                                transforms=transform)
    return dataset_train


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
