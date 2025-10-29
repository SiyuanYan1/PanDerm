# import torchtoolbox.transform as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Derm_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, root, train=False,val=False,test=False,transforms=None,binary=False,data_percent=1):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age

		"""
        if train==True:
            self.df = df[df['split']=='train']
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            half_rows = int(len(self.df) * data_percent)
            self.df=self.df.head(half_rows)
        elif val==True:
            self.df = df[df['split'] == 'val']
        elif test==True:
            self.df = df[df['split'] == 'test']
            
        self.transforms = transforms
        self.root=root
        self.binary=binary

    # import torchsnooper
    # @torchsnooper.snoop()
    def __getitem__(self, index):
        filename = self.df.iloc[index]['image']
        # im_path = '/home/share/Eval_Data/ISIC_2019_Training/'+filename+'.jpg'

        im_path = str(self.root)+str(filename)
        # Use PIL to load the image directly in RGB format
        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        # print(x)
        # Apply transformations if any
        if x is not None and self.transforms:
            x = self.transforms(x)
            y = self.df.iloc[index]['binary_label'] if self.binary else self.df.iloc[index]['label']
        return x,y,filename

    def __len__(self):
        return len(self.df)

class Uni_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, root, train=False,val=False,test=False,transforms=None,binary=False,data_percent=1, image_key='image'):
        """
		This dataset is used for finetuning only

		"""
        if train==True:
            self.df = df[df['split']=='train']
            half_rows = int(len(self.df) * data_percent)
            self.df=self.df.head(half_rows)
        elif val==True:
            self.df = df[df['split'] == 'val']
        elif test==True:
            self.df = df[df['split'] == 'test']
        else:
            self.df = df
        self.transforms = transforms
        self.root=root
        self.binary=binary
        self.image_key = image_key

    def __getitem__(self, index):
        filename = self.df.iloc[index][self.image_key]
        im_path = str(self.root) + str(filename)

        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        if x is not None and self.transforms:
            x = self.transforms(x)

        if self.binary:
            y = self.df.iloc[index]['binary_label']
        else:
            y = self.df.iloc[index]['label']

        # Return the filename along with the image and label
        return x, filename, y

    def __len__(self):
        return len(self.df)


    def count_label(self):
        label_counts = self.df['label'].value_counts().sort_index()
        return label_counts