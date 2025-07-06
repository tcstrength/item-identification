import random
from torchvision import transforms as T

def get_transforms_v1():
    transform_train = T.Compose([
        T.RandomResizedCrop(
            size=224,                # output size (height and width)
            scale=(0.8, 1.0),       # crop area range (8% to 100% of original)
            ratio=(0.75, 1.3333)     # aspect ratio range
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.3, 0.3),  # 5% translation in both directions
            shear=30             # shear angle
        ),

        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_v2():
    transform_train = T.Compose([
        T.RandomResizedCrop(
            size=224,                # output size (height and width)
            scale=(0.8, 1.0),       # crop area range (8% to 100% of original)
            ratio=(0.75, 1.3333)     # aspect ratio range
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.3, 0.3),  # 5% translation in both directions
            shear=30             # shear angle
        ),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_pure():
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_downscale(size: int=64):
    transform_train = T.Compose([
        T.Resize((size, size)),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize((size, size)),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_downscale_random(size: int=128):
    transform_train = T.Compose([
        T.Lambda(lambda img: T.Resize(random.randint(64, 196))(img)),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize((size, size)),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_downscale_random_clip():
    transform_train = T.Compose([
        T.Lambda(lambda img: T.Resize(random.randint(32, 224))(img)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.2),
            ratio=(0.75, 1.3333)
        )
    ])

    transform_test = None

    return transform_train, transform_test

def get_transforms_downscale_random_v2(size: int = 128):
    transform_train = T.Compose([
        T.Lambda(lambda img: T.Resize(random.randint(32, 224))(img)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.2),
            ratio=(0.75, 1.3333)
        ),
        T.ToTensor()
    ])

    transform_test = T.Compose([
        T.Resize((size, size)),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test

def get_transforms_downscale_random_v2(size: int = 128):
    transform_train = T.Compose([
        T.Lambda(lambda img: T.Resize(random.randint(32, 224))(img)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=90,          # additional rotation control
            translate=(0.2, 0.2),  # 5% translation in both directions
            shear=10             # shear angle
        ),
        T.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.2),
            ratio=(0.75, 1.3333)
        ),
        T.ToTensor()
    ])

    transform_test = T.Compose([
        T.Resize((size, size)),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    return transform_train, transform_test
