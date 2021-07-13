import os
import shutil
from torchvision.io import read_image
from tqdm import tqdm
import psutil
import ray
path = '/home/user/data/CC12M/images'
broken_path = '/home/user/data/CC12M/broken_images'
num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)
@ray.remote
def check_image(image_file):
    image_path = os.path.join(path, image_file)
    try:
        img = read_image(image_path)
    except:
        shutil.copy(image_path, os.path.join(broken_path, image_file))
        shutil.remove(image_path)


for idx, image_file in enumerate(os.listdir(path)):
    check_image.remote(image_file)
    print(idx)