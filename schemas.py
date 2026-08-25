from typing import Optional, List
from pydantic import BaseModel


class BuiltinDatasetReq(BaseModel):
    name: str  # cifar10 | cifar100 | mnist | fashionmnist
    img_size: Optional[int] = None
    batch_size: int = 64


class KaggleDatasetReq(BaseModel):
    slug: str
    username: str
    key: str
    img_size: int = 128
    batch_size: int = 32
    val_split: float = 0.2


class UrlItem(BaseModel):
    url: str
    label: str


class UrlDatasetReq(BaseModel):
    items: List[UrlItem]
    img_size: int = 128
    batch_size: int = 16
    val_split: float = 0.2


class ModelSelectReq(BaseModel):
    name: str
    pretrained: bool = False


class ModelEnhanceReq(BaseModel):
    kind: str  # none | spp | adaptive_ppm | feature_denoise


class TrainCleanReq(BaseModel):
    epochs: int = 5
    lr: float = 1e-3


class TrainAdvReq(BaseModel):
    method: str  # fgsm_at | pgd_at | trades | awp_pgd_at
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 7
    epochs: int = 5
    lr: float = 1e-3
    beta: float = 6.0          # TRADES only
    awp_gamma: float = 0.01    # AWP only


class AttackReq(BaseModel):
    attack: str  # fgsm | ifgsm | pgd | autoattack | square | transfer
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 10
    n_samples: int = 500


class GradCamReq(BaseModel):
    n_samples: int = 8
    mode: str = "clean"  # clean | adversarial
    attack: Optional[str] = "fgsm"
    epsilon: float = 8 / 255


class SmoothingReq(BaseModel):
    sigma: float = 0.25
    n_noise_samples: int = 50
    max_batches: int = 10


class PipelineStep(BaseModel):
    type: str
    params: dict = {}


class PipelineReq(BaseModel):
    steps: List[PipelineStep]
