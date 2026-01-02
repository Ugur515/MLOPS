from dagster import asset, Definitions, AssetExecutionContext, Field, Output, MetadataValue, AssetMaterialization
from dagster_mlflow import mlflow_tracking
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import os
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuration ----------------
NOTEBOOK_DIR = Path().resolve()
DATA_CSV = NOTEBOOK_DIR / "2025_10_13_mlops_biomass_data/mlops_biomass_data/digital_biomass_labels.xlsx"
IMAGES_DIR = NOTEBOOK_DIR / "2025_10_13_mlops_biomass_data/mlops_biomass_data/images_med_res"

FIGURES_DIR = NOTEBOOK_DIR / "figures"
CHECKPOINTS_DIR = NOTEBOOK_DIR / "checkpoints"
RESULTS_DIR = NOTEBOOK_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

VAL_FRACTION = 0.2
NUM_WORKERS = 0


# ---------------- Utility / Dataset ----------------
class PlantBiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None, filename_col="filename", target_col="fresh_weight_total"):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.filename_col = filename_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[self.filename_col]
        label = row[self.target_col]
        img_path = self.image_dir / str(fname)

        if not img_path.exists():
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return image, label_tensor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def build_model(model_name: str):
    if model_name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet50'")
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, 1)
    return base


# ---------------- Assets ----------------
@asset
def raw_dataset(context: AssetExecutionContext):
    context.log.info("raw_dataset: Lade Metadaten...")

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"metadata file not found: {DATA_CSV}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"images directory not found: {IMAGES_DIR}")

    df = pd.read_excel(DATA_CSV)
    df_clean = df.dropna(subset=["fresh_weight_total", "filename"]).copy()

    df_clean["image_exists"] = df_clean["filename"].apply(lambda fn: (IMAGES_DIR / str(fn)).exists())
    df_clean = df_clean[df_clean["image_exists"]].drop(columns=["image_exists"]).reset_index(drop=True)
    image_paths = [(IMAGES_DIR / str(f)).as_posix() for f in df_clean["filename"].tolist()]

    context.log.info(f"raw_dataset: Loaded {len(df_clean)} valid samples.")
    return {"images": image_paths, "metadata": df_clean}


@asset
def eda_plots(context: AssetExecutionContext, raw_dataset):
    df = raw_dataset["metadata"]
    saved = {}

    try:
        plt.figure(figsize=(8, 5))
        plt.hist(df["fresh_weight_total"].dropna(), bins=50)
        plt.title("Target Distribution")
        path = FIGURES_DIR / "target_distribution.png"
        plt.savefig(path)
        plt.close()
        saved["target_distribution"] = str(path)
    except Exception as e:
        context.log.error(f"Error target_distribution: {e}")

    return {"figures": saved}


@asset
def preprocessed_data(context: AssetExecutionContext, raw_dataset):
    df = raw_dataset["metadata"]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    total = len(df)
    val_len = int(total * VAL_FRACTION)
    train_len = total - val_len

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuffled.iloc[:train_len].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_len:].reset_index(drop=True)

    train_dataset = PlantBiomassDataset(train_df, IMAGES_DIR, transform=train_transforms)
    val_dataset = PlantBiomassDataset(val_df, IMAGES_DIR, transform=val_transforms)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset
    }


# ---------------- TRAINING WITH CONFIG ----------------
@asset(
    required_resource_keys={"mlflow"},
    config_schema={
        "epochs": Field(int, default_value=3),
        "batch_size": Field(int, default_value=32),
        "learning_rate": Field(float, default_value=1e-3),
        "model_name": Field(str, default_value="resnet18")
    }
)
def trained_model(context: AssetExecutionContext, preprocessed_data):

    cfg = context.op_config
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    model_name = cfg["model_name"]

    context.log.info(f"Training-Config: epochs={epochs}, bs={batch_size}, lr={lr}, model={model_name}")

    train_loader = DataLoader(
        preprocessed_data["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        preprocessed_data["val_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = get_device()
    model = build_model(model_name).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_checkpoint = CHECKPOINTS_DIR / "best_model_mlflow.pth"
    final_model_path = CHECKPOINTS_DIR / "final_model_mlflow.pth"
    best_val_loss = float("inf")
    # SSE Total für R2 vorab berechnen
    all_val_labels = []
    for _, labels in val_loader:
        all_val_labels.append(labels)
    if len(all_val_labels) > 0:
        all_val_labels = torch.cat(all_val_labels).squeeze(1)
        sse_total = torch.sum((all_val_labels - torch.mean(all_val_labels)) ** 2).item()
    else:
        sse_total = 0.0
    with mlflow.start_run(run_name="dagster_resnet_training", nested=True):

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": model_name,
            "optimizer": "Adam"
        })

        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0
            train_count = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * imgs.size(0)
                train_count += imgs.size(0)

            train_loss = train_loss_sum / max(train_count, 1)

            # validation
            model.eval()
            val_loss_sum = 0
            val_count = 0
            sse_residual = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * imgs.size(0)
                    sse_residual += torch.sum((labels - outputs) ** 2).item()
                    val_count += imgs.size(0)

            val_loss = val_loss_sum / max(val_count, 1)
            epoch_r2 = 1 - (sse_residual / sse_total) if sse_total > 1e-8 else float("nan")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_r2": epoch_r2
            }, step=epoch)

            context.log.info(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_checkpoint)

        torch.save(model.state_dict(), final_model_path)

        if best_checkpoint.exists():
            mlflow.log_artifact(str(best_checkpoint), artifact_path="models")

    return {
        "checkpoint_path": str(best_checkpoint),
        "final_model_path": str(final_model_path)
    }



def _img_to_md(path: Path, title: str) -> MetadataValue:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return MetadataValue.md(f"![{title}](data:image/png;base64,{encoded})")


@asset
def model_evaluation(context: AssetExecutionContext, trained_model, preprocessed_data):
    checkpoint = Path(trained_model["checkpoint_path"])
    device = get_device()

    model = build_model("resnet18").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    val_loader = DataLoader(preprocessed_data["val_dataset"], batch_size=32, shuffle=False)

    preds, gts = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds.extend(outputs.cpu().numpy().flatten())
            gts.extend(labels.cpu().numpy().flatten())

    preds = np.array(preds)
    gts = np.array(gts)
    mse = np.mean((gts - preds) ** 2)

    # Plot speichern
    plot_path = RESULTS_DIR / "pred_vs_actual.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gts, preds, s=6)
    ax.plot([gts.min(), gts.max()], [gts.min(), gts.max()], "--")
    ax.set_title(f"Pred vs Actual (MSE={mse:.2f})")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # Metadata als Markdown-Bild
    plot_md = _img_to_md(plot_path, "Predictions vs Actual")

    context.log.info(f"Plot gespeichert unter {plot_path}")

    return Output(
        value={"val_mse": float(mse)},
        metadata={
            "evaluation_plot": plot_md
        }
    )


# ---------------- Definitions ----------------
defs = Definitions(
    assets=[raw_dataset, eda_plots, preprocessed_data, trained_model, model_evaluation],
    resources={
        "mlflow": mlflow_tracking.configured({
            "experiment_name": "plant_biomass_pipeline",
            "mlflow_tracking_uri": "http://localhost:5001"
        })
    }
)