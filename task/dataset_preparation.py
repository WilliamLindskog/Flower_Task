from torchvision.transforms import Compose, Normalize, ToTensor

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    # Pytorch transforms for MNIST
    pytorch_transforms = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ]
    )
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch