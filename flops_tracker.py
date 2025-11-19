from __future__ import annotations

from typing import Optional, Any

from .tracker import Tracker


class FlopsTracker:
    """
    Entry point pubblico della libreria.

    Uso:
        ft = FlopsTracker(run_name="dp_cnn").torch_bind(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            device=device,
            epochs=10,
            log_per_batch=True,
            export_path="flops.csv",
            use_wandb=False,
        )
    """

    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name
        self._raw_flops: int = 0
        self._total_flops: float = 0.0 
        self._history: dict[str, Any] = {}

    @property
    def raw_flops(self) -> int:
        """FLOPs grezzi contati dal backend (somma dei layer)."""
        return self._raw_flops

    @property
    def total_flops(self) -> float:
        """
        FLOPs totali "esposti" all'utente.
        Per ora coincidono con i raw FLOPs,
        ma puoi usare questo per eventuali correzioni future.
        """
        return self._total_flops

    @property
    def history(self) -> dict[str, Any]:
        return self._history

    # -------------------- SKLEARN BIND -------------------- #

    def sklearn_bind(
        self,
        model,
        X,
        y=None,
        *,
        mode: str = "fit",  # "fit", "predict" o "fit_predict"
        backend: str = "sklearn",
        log_per_call: bool = True,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopsTracker":
        """
        Esegue fit/predict per un modello sklearn e traccia i FLOPs.

        Esempi:
            ft = FlopsTracker(run_name="lr").sklearn_bind(
                model=clf,
                X=X_train,
                y=y_train,
                mode="fit",
                log_per_call=True,
                export_path="flops_lr.csv",
            )

            ft = FlopsTracker(run_name="knn").sklearn_bind(
                model=knn,
                X=X_test,
                mode="predict",
            )

        Ogni chiamata a fit/predict viene trattata come un "batch"
        nei log del backend sklearn.
        """

        from .tracker import Tracker

        log_per_batch = log_per_call
        log_per_epoch = False  
        with Tracker(
            model=model,
            backend=backend,
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            # eseguiamo fit/predict secondo la modalità scelta
            if mode == "fit":
                model.fit(X, y)
            elif mode == "predict":
                _ = model.predict(X)
            elif mode == "fit_predict":
                model.fit(X, y)
                _ = model.predict(X)
            else:
                raise ValueError(f"Modo sklearn_bind non supportato: {mode}")

            self._raw_flops = tr.total_flops
            self._total_flops = float(self._raw_flops)

            self._history["backend"] = backend
            self._history["export_path"] = export_path
            self._history["use_wandb"] = use_wandb
            self._history["wandb_project"] = wandb_project
            self._history["mode"] = mode

        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        print(
            f"[FlopsTracker{run_label}] FLOPs totali (sklearn, mode={mode}): "
            f"{self._total_flops:.0f} (raw: {self._raw_flops})"
        )

        return self


    # -------------------- TORCH BIND -------------------- #

    def torch_bind(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        device: Optional[str] = None,
        *,
        epochs: int = 1,
        backend: str = "torch",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopsTracker":
        """
        Esegue training + tracking FLOPs per un modello PyTorch.
        questa funzione incapsula loop per batch/epoch e print dei FLOPs totatli (totalmente trasparente per l'utente).
        """

        if device is not None:
            model.to(device)

        from .tracker import Tracker

        with Tracker(
            model=model,
            backend=backend,
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            for epoch in range(epochs):
                if hasattr(tr.backend, "set_epoch"):
                    tr.backend.set_epoch(epoch)

                for xb, yb in train_loader:
                    if device is not None:
                        xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    output = model(xb)

                    if loss_fn is not None:
                        loss = loss_fn(output, yb)
                        loss.backward()
                        optimizer.step()
                    
                # log per epoch (se abilitato)
                if tr.logger is not None and hasattr(tr.logger, "log_epoch"):
                    tr.logger.log_epoch(
                        epoch=epoch,
                        flops=tr.total_flops,
                        cumulative_flops=tr.total_flops,
                    )

            
            self._raw_flops = tr.total_flops
            self._total_flops = float(self._raw_flops)

            self._history["backend"] = backend
            self._history["export_path"] = export_path
            self._history["use_wandb"] = use_wandb
            self._history["wandb_project"] = wandb_project
            self._history["epochs"] = epochs

        # STAMPA AUTOMATICA DEI FLOPs TOTALI
        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        print(
            f"[FlopsTracker{run_label}] FLOPs totali: {self._total_flops:.0f} "
            f"(raw: {self._raw_flops})"
        )

        return self
