from monai.transforms import RandGaussianNoise, SqueezeDim, Rand3DElastic, RandGaussianSmooth, RandZoom, RandAffine, LoadImage, AddChannel, ToTensor, Resize, NormalizeIntensity, RandRotate90, RandFlip
from monai.transforms import Compose, Rand3DElasticd, RandGaussianNoised, SqueezeDimd, \
                             LoadImaged, AddChanneld, ToTensord, Resized, NormalizeIntensityd, \
                             RandRotate90d, RandFlipd, ConcatItemsd, RandZoomd, RandGaussianSmoothd, \
                             RandAffined

transform_train_mri = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        Resize(spatial_size=(128, 128, 128)),
        NormalizeIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandGaussianNoise(prob=0.5),
        ToTensor()
    ]
)

transform_val_mri = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        Resize(spatial_size=(128, 128, 128)),
        NormalizeIntensity(),
        ToTensor()
    ]
)

transform_train_pet = Compose(
    [
        LoadImage(image_only=True),
        # SqueezeDim(dim=3),
        AddChannel(),
        # Resize(spatial_size=(128, 128, 128)),
        NormalizeIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandAffine(prob=0.5),
        Rand3DElastic(sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.5,
                      padding_mode='zeros'),
        RandZoom(prob=0.5, min_zoom=0.6, max_zoom=0.8),
        RandGaussianNoise(prob=0.5),
        RandGaussianSmooth(prob=0.5, sigma_x=(1, 2)),
        ToTensor()
    ]
)

transform_val_pet = Compose(
    [
        LoadImage(image_only=True),
        # SqueezeDim(dim=3),
        AddChannel(),
        # Resize(spatial_size=(128, 128, 128)),
        NormalizeIntensity(),
        ToTensor()
    ]
)

fusionTransform_train = Compose(
    [
        LoadImaged(keys=["mri", "pet"], image_only=True),
        SqueezeDimd(keys=["pet"], dim=3),
        AddChanneld(keys=["mri", "pet"]),
        NormalizeIntensityd(keys=["mri", "pet"]),
        Resized(keys=["mri", "pet"], spatial_size=(128, 128, 128)),
        RandRotate90d(keys=["mri", "pet"], prob=0.5),
        RandFlipd(keys=["mri", "pet"], prob=0.5),
        RandAffined(keys=["mri", "pet"], prob=0.5,
                    shear_range=(0.5, 0.5),
                    mode=['bilinear', 'nearest'],
                    padding_mode='zeros'),
        Rand3DElasticd(keys=["mri", "pet"], prob=0.5,
                        sigma_range=(5, 7), magnitude_range=(50, 150),
                        padding_mode='zeros'),
        RandZoomd(keys=["mri", "pet"], prob=0.5,
                  min_zoom=1.3, max_zoom=1.5),
        RandGaussianNoised(keys=["mri", "pet"], prob=0.5),
        RandGaussianSmoothd(keys=["mri", "pet"], prob=0.5),
        ToTensord(keys=["mri", "pet"])
    ]
)

fusionTransform_val = Compose(
    [
        LoadImaged(keys=["mri", "pet"], image_only=True),
        SqueezeDimd(keys=["pet"], dim=3),
        AddChanneld(keys=["mri", "pet"]),
        NormalizeIntensityd(keys=["mri", "pet"]),
        Resized(keys=["mri", "pet"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["mri", "pet"])
    ]
 )

