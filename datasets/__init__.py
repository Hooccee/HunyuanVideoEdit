from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .V2VBench import V2VBench_dataset



def get_dataloader(dataset_name, video_size=None, video_length=None, rescale=True, **kwargs):
    if dataset_name == 'V2VBench':
        return V2VBench_dataset(
            video_size=video_size,
            video_length=video_length,
            rescale=rescale,
            **kwargs
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")