import re
import monai
import os
import numpy as np
from glob import glob
from monai.data import Dataset
import transform

def Multi_modal_data(mri_dir, pet_dir):
    mri_dir = mri_dir + '/*.nii'
    pet_dir = pet_dir + '/*.nii'
    mri_list = sorted(glob(mri_dir))
    pet_list = sorted(glob(pet_dir))
    mri_list.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    pet_list.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    label_dic = {'CN': 0, 'AD': 1}
    label = []
    number = 0
    for i in enumerate(mri_list):
        label.insert(number, [label_dic[str(i).split('_')[-2]]])
        number = number + 1
    label = np.array(label)
    data_dict = [{'mri': mri, 'pet': pet, 'label': label} for mri, pet, label in zip(mri_list, pet_list, label)]
    return data_dict

if __name__ == '__main__':
    data_dict = Multi_modal_data('/Library/GM/Dataset_preprocessed/test/MRI', '/Library/GM/Dataset_preprocessed/test/PET')
    print(data_dict[:2])
    print(data_dict[-2:])
    # # label = DataList.rtList(image_dir='/Library/GM/Dataset_new/', stage='test').image_label
    dataset = Dataset(data=data_dict[:5], transform=transform.fusionTransform_val)
    dataloader = monai.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for data in dataloader:
        x, y = data["image"], data["label"]
        print(x.shape, y.shape)
        break
