import csv
import os
import tempfile


def append_loss_to_csv(epoch_log, csv_path="training_log.csv"):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_z_proprio_loss",
        "train_standard_l2_loss",
        "train_bisim_loss",
        "train_bisim_z_dist",
        "train_bisim_r_dist",
        "train_bisim_var_loss",
        "train_bisim_transition_dist",
        "train_bisim_cov_reg",
        "train_bisim_vicreg_inv",
        "train_bisim_vicreg_total",
        "train_dinowm_z_loss",
        "val_dinowm_z_loss",
        "val_z_proprio_loss",
        "val_standard_l2_loss",
        "val_bisim_loss",
        "val_bisim_z_dist",
        "val_bisim_r_dist",
        "val_bisim_var_loss",
        "val_bisim_transition_dist",
        "val_bisim_cov_reg",
        "val_bisim_vicreg_inv",
        "val_bisim_vicreg_total",
    ]

    new_row = {k: epoch_log.get(k) for k in fieldnames}
    new_row["epoch"] = epoch_log.get("epoch")

    rows = []
    if os.path.isfile(csv_path):
        try:
            raw = open(csv_path, "rb").read().replace(b"\x00", b"")
            text = raw.decode("utf-8", errors="replace")
            reader = csv.DictReader(text.splitlines())
            for row in reader:
                if not row or not any(row.values()):
                    continue
                rows.append({k: row.get(k) for k in fieldnames})
        except (OSError, csv.Error, UnicodeError):
            rows = []

    # Keep one row per epoch (last write wins) so multi-rank races don't duplicate.
    by_epoch = {}
    for row in rows:
        ep = row.get("epoch")
        if ep is None or ep == "":
            continue
        by_epoch[str(ep)] = row
    ep_new = new_row.get("epoch")
    if ep_new is not None and ep_new != "":
        by_epoch[str(ep_new)] = new_row

    def _ep_key(ep):
        try:
            return int(float(ep))
        except (TypeError, ValueError):
            return ep

    ordered = [by_epoch[k] for k in sorted(by_epoch.keys(), key=_ep_key)]

    # Atomic replace avoids truncated/NUL files when multiple writers collide.
    dirname = os.path.dirname(os.path.abspath(csv_path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".training_loss_", suffix=".csv", dir=dirname)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
