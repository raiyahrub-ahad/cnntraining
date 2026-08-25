"""Dataset ingestion. Supports:
  - built-in torchvision datasets (CIFAR10/100, MNIST, FashionMNIST) for quick
    experimentation
  - Kaggle datasets via the Kaggle API (requires username + key)
  - a list of direct image URLs with labels
  - an uploaded zip of a class-per-folder image dataset

All pipelines output plain ToTensor (range [0,1], no Normalize) so that
adversarial attacks operate in a well-defined epsilon-ball directly on pixel
values, which is the convention `torchattacks` expects.
"""

import os
import shutil
import urllib.request
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

DATA_ROOT = os.environ.get("PHD_DATA_ROOT", os.path.join(os.getcwd(), "data_cache"))
os.makedirs(DATA_ROOT, exist_ok=True)

BUILTIN_DATASETS = {
    "cifar10": {"cls": torchvision.datasets.CIFAR10, "classes": 10, "channels": 3, "size": 32},
    "cifar100": {"cls": torchvision.datasets.CIFAR100, "classes": 100, "channels": 3, "size": 32},
    "mnist": {"cls": torchvision.datasets.MNIST, "classes": 10, "channels": 1, "size": 28},
    "fashionmnist": {"cls": torchvision.datasets.FashionMNIST, "classes": 10, "channels": 1, "size": 28},
}


def load_builtin(name, img_size=None, batch_size=64):
    name = name.lower()
    info = BUILTIN_DATASETS[name]
    size = img_size or info["size"]
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    train_ds = info["cls"](root=DATA_ROOT, train=True, download=True, transform=tf)
    test_ds = info["cls"](root=DATA_ROOT, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, info["classes"], info["channels"], size


def load_image_folder(path, img_size=128, batch_size=32, val_split=0.2):
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    full_ds = torchvision.datasets.ImageFolder(root=path, transform=tf)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, test_ds = random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, len(full_ds.classes), 3, img_size


def extract_zip_dataset(zip_path, extract_to):
    import zipfile
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    entries = [e for e in os.listdir(extract_to) if not e.startswith("__MACOSX")]
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_to, entries[0])):
        return os.path.join(extract_to, entries[0])
    return extract_to


def download_kaggle_dataset(slug, username, key, dest_dir):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest_dir, exist_ok=True)
    api.dataset_download_files(slug, path=dest_dir, unzip=True)
    entries = [e for e in os.listdir(dest_dir) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest_dir, entries[0])):
        return os.path.join(dest_dir, entries[0])
    return dest_dir


def download_images_from_urls(url_label_pairs, dest_dir):
    """url_label_pairs: [{'url':..., 'label':...}, ...]. Downloads into
    dest_dir/<label>/<n>.jpg so the result is ImageFolder-compatible."""
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    counters = {}
    for item in url_label_pairs:
        label = item.get("label", "unlabeled")
        url = item["url"]
        label_dir = os.path.join(dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        counters[label] = counters.get(label, 0) + 1
        out_path = os.path.join(label_dir, f"{counters[label]}.jpg")
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return dest_dir
