from torchvision import transforms

def train_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    return transforms.Compose(transform_list)

def test_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    return transforms.Compose(transform_list)