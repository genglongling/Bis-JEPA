import csv
import os


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
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                rows.append({k: row.get(k) for k in fieldnames})
    rows.append(new_row)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
