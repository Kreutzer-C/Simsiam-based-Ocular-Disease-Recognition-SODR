import torchvision.transforms as T

class SimSiamTransform():
    def __init__(self):
        image_size = 224   # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0
        mean_std = [[0.2913,0.1849,0.1014],[0.3103,0.2117,0.1381]] # ODIR-5K
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

class Transform_single():
    def __init__(self):
        image_size = 224
        p_blur = 0.5
        mean_std = [[0.2913, 0.1849, 0.1014], [0.3103, 0.2117, 0.1381]]  # ODIR-5K
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        return self.transform(x)

class Transform_test():
    def __init__(self):
        image_size = 224
        mean_std = [[0.2913, 0.1849, 0.1014], [0.3103, 0.2117, 0.1381]]  # ODIR-5K
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        return self.transform(x)