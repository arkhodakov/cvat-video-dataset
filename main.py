from datagen import VideosGenerator

from decord import cpu
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    root = Path().resolve().parent.parent.joinpath("data")
    dataset = VideosGenerator(root)
    print("Length: ", len(dataset))
    for index in tqdm(range(len(dataset))):
        batch = dataset[index]
