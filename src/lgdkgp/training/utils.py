import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_fit_start(self, trainer, pl_module):
        # stable-pretraining can redirect Lightning checkpoint/log directories.
        import json
        from pathlib import Path
        from stable_worldmodel.data import get_cache_dir
        if str(self.run_name).startswith("kernel_prior_"):
            root = Path(get_cache_dir(sub_folder="checkpoints")) / str(self.run_name).split("/")[0]
            root.mkdir(parents=True, exist_ok=True)
            paths = [str(cb.dirpath) for cb in trainer.checkpoint_callbacks if cb.dirpath]
            (root / "runtime.json").write_text(json.dumps({"checkpoint_dirs": paths, "log_dir": trainer.logger.log_dir}, indent=2))

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
        # Retain only the latest epoch weights; optimizer resume lives separately.
        if str(self.run_name).startswith("kernel_prior_"):
            from pathlib import Path
            from stable_worldmodel.data import get_cache_dir
            folder = Path(get_cache_dir(sub_folder="checkpoints")) / self.run_name
            for old in folder.glob("weights_epoch_*.pt"):
                if old.name != f"weights_epoch_{epoch}.pt":
                    old.unlink()


class RunIdentityCallback(Callback):
    """Bind optimizer checkpoints and fit-end evidence to an immutable run."""

    def __init__(self, identity_sha256, evidence_path):
        self.identity_sha256 = identity_sha256
        self.evidence_path = evidence_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['lewm_run_identity'] = {
            'identity_sha256': self.identity_sha256,
            'max_epochs': trainer.max_epochs,
        }

    def on_fit_end(self, trainer, pl_module):
        from lgdkgp.common.run_identity import write_json
        write_json(self.evidence_path, {
            'identity_sha256': self.identity_sha256,
            'max_epochs': trainer.max_epochs,
            'completed_epochs': trainer.current_epoch,
            'global_step': trainer.global_step,
            'validation_metrics': {k: float(v) for k, v in trainer.callback_metrics.items()
                                   if k.startswith(('val/', 'validate/')) and v.numel() == 1},
        })
