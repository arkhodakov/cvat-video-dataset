import math
import decord
import numpy as np
import xmltodict

from collections import defaultdict

from decord._ffi.ndarray import DECORDContext
from decord import VideoReader

from typing import Any, Dict, List, OrderedDict, Tuple
from pathlib import Path


class VideosGenerator():
    """ Custom data generator with the support of CVAT annotated videos.

        Arguments:
            `root`: str - path to dataset root directory.
            `shape`: tuple - (width, height, chanels) of an image to resize (Default: (256, 256, 3)).
            `batch_size`: int - number of images/boxes in batches (Default: 32).
            `decoding_context`: DECORDContext - decord device for video decoding (Default: cpu).
            `in_memory_decoding`: bool - load source videos to RAM with `open` (Default: False).
    """
    def __init__(
        self,
        root: Path,
        shape: Tuple = (256, 256, 3),
        batch_size: int = 32,
        decoding_context: DECORDContext = decord.cpu(0),
        in_memory_decoding: bool = False
    ) -> None:
        self.root: Path = root
        self.shape: Tuple = shape
        self.batch_size: int = batch_size
        self.decoding_context: DECORDContext = decoding_context
        self.in_memory_decoding: bool = in_memory_decoding

        if not list(root.glob("*.xml")):
            raise RuntimeError(f"Couldn't find any .xml annotations in the root directory: {self.root}")

        decord.bridge.set_bridge('torch')

        self.dataset: List[List] = []
        self.readers: List[VideoReader] = []

        for index, annotation_path in enumerate(root.glob("*.xml")):
            with open(annotation_path, "r", encoding="utf-8") as file:
                annotation = xmltodict.parse(file.read(), encoding="utf-8")["annotations"]
            source = self.root.joinpath(annotation["meta"]["source"])
            if not source.exists():
                raise RuntimeError(f"Couldn't find the source video: {source}")
            source = open(source, "rb") if in_memory_decoding else str(source)
            boxes = self.__format_tracks(annotation["track"], index)
            self.dataset.extend(boxes)
            reader = VideoReader(source, ctx=self.decoding_context, width=shape[1], height=shape[0])
            self.readers.append(reader)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        readersmap = defaultdict(list)
        for item in sample:
            source_index, frame = item[0], item[5]
            readersmap[source_index].append(frame)

        framesmap = {}
        for i, (index, frames) in enumerate(readersmap.items()):
            images = self.readers[index].get_batch(frames)
            framesmap[i] = dict(zip(frames, images))

        images, boxes, meta = [], [], []
        for (source_index, x1, y1, x2, y2, frame, id, label) in sample:
            images.append(np.array(framesmap[source_index][frame], dtype=np.uint8))
            boxes.append((x1, y1, x2, y2))
            meta.append((id, label))
        images = np.array(images, dtype=np.uint8)
        boxes = np.array(boxes, dtype=np.float32)
        meta = np.array(meta, dtype=object)
        return {"images": images, "boxes": boxes, "meta": meta}

    def __format_tracks(self, track: Any, source_index: int) -> List:
        """ Parse XML track node appending source index at the beginning of boxes arrays."""
        boxes: List[Dict] = []
        if isinstance(track, List):
            for track in [dict(track) for track in track]:
                for box in [dict(box) for box in track["box"]]:
                    box.update({
                        "id": track["@id"],
                        "label": track["@label"]
                    })
                    boxes.append(box)
        elif isinstance(track, OrderedDict):
            track = dict(track)
            for box in [dict(box) for box in track["box"]]:
                box.update({
                    "id": track["@id"],
                    "label": track["@label"]
                })
                boxes.append(box)

        for i, box in enumerate(boxes):
            id, label = int(box["id"]), box["label"]
            x1, y1, x2, y2 = float(box["@xtl"]), float(box["@ytl"]), float(box["@xbr"]), float(box["@ybr"])
            frame = int(box["@frame"])
            boxes[i] = [source_index, x1, y1, x2, y2, frame, id, label]
        return boxes

    def on_epoch_end(self) -> None:
        """ Tensorflow: Method called at the end of every epoch.
            Docs: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence#on_epoch_end."""
        indices = np.random.permutation(len(self.dataset))
        self.dataset = self.dataset[indices]
