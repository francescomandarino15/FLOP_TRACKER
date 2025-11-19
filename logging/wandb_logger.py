from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str | None,
        token: str | None,
        log_per_batch: bool,
        log_per_epoch: bool,
        run_name: str | None = None,
    ):
        super().__init__(log_per_batch=log_per_batch, log_per_epoch=log_per_epoch)
        self._wandb = None
        self.run = None

        if project is not None:
            import wandb
            if token is not None:
                wandb.login(key=token)
            self._wandb = wandb
            self.run = wandb.init(project=project, name=run_name)

    def log_batch(self, step, flops, cumulative_flops, epoch=None):
        if not self.log_per_batch or self._wandb is None:
            return
        self._wandb.log(
            {
                "flops_batch": flops,
                "flops_cumulative": cumulative_flops,
                "batch_step": step,
                "epoch": epoch,
            }
        )

    def log_epoch(self, epoch, flops, cumulative_flops):
        if not self.log_per_epoch or self._wandb is None:
            return
        self._wandb.log(
            {
                "flops_epoch": flops,
                "flops_cumulative": cumulative_flops,
                "epoch": epoch,
            }
        )

    def close(self):
        if self.run is not None:
            self.run.finish()
