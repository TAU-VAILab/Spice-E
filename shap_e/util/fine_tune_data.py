import os
import json
import glob
import torch
import random
from torch.utils.data import Dataset


class ShapE_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        load_gray: bool = False,
        refined_ver: str = "v1",
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.refined_ver = refined_ver
        self.shape_paths = glob.glob(f"{root_path}/*", recursive=False)
        self.load_gray = load_gray
        self.shape_paths = [
            shape_path
            for shape_path in self.shape_paths
            if os.path.exists(os.path.join(shape_path, "latent.pt"))
        ]
        if self.load_gray:
            self.shape_paths = [
                shape_path
                for shape_path in self.shape_paths
                if os.path.exists(os.path.join(shape_path, "latent_gray.pt"))
            ]
        print("dataset length: ", len(self))

    def __len__(self):
        return len(self.shape_paths)

    def get_possible_item(self, index: int):
        shapepath = self.shape_paths[index]
        latent_path = os.path.join(shapepath, "latent.pt")
        gray_latent_path = os.path.join(shapepath, "latent_gray.pt")
        annotation_path = os.path.join(shapepath, "annotation.json")

        # load latent and annotation
        latent = torch.load(latent_path)
        annotation_json = json.load(open(annotation_path, "r"))
        if f"refind_annotation_{self.refined_ver}" not in annotation_json.keys():
            if self.verbose:
                print(f"refined annotation {self.refined_ver} not found")
            shape_description = annotation_json["name"]
        else:
            shape_description = annotation_json[f"refind_annotation_{self.refined_ver}"]
        sample = {"latent": latent, "prompt": shape_description}
        if self.load_gray:
            gray_latent = torch.load(gray_latent_path)
            sample["latent_gray"] = gray_latent

        return sample

    def __getitem__(self, index):
        success = False
        try:
            example = self.get_possible_item(index)
            success = True
        except Exception as e:
            print("!!!error ", index, e)
            index = random.randint(0, len(self))
            return -1
        return example
