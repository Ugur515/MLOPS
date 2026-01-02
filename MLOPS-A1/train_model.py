# %%
# ========================================================================
# IMPORTE
# ========================================================================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import random
import os
import time
import math
import torch.nn as nn
import torchvision.models as models

# NEU für Schritt 4:
import argparse
import logging
import subprocess
import sys

# %%
# ========================================================================
# SCHRITT 4.A: ARGPARSE UND LOGGING SETUP
# ========================================================================

# === 4.A.1: Argument Parser ===
parser = argparse.ArgumentParser(description="MLOps Lab 1: Biomasse-Modell trainieren")

parser.add_argument('--epochs', type=int, default=20,
                    help='Anzahl der Trainingsepochen (default: 20)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch-Größe für Training und Validierung (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Lernrate für den Adam-Optimizer (default: 0.001)')
parser.add_argument('--model_name', type=str, default='resnet18',
                    choices=['resnet18', 'resnet50'], 
                    help='Zu verwendendes ResNet-Modell: resnet18 or resnet50 (default: resnet18)')

# Parsen der Argumente (wird innerhalb von main() aufgerufen)


# === 4.A.2: Logging Setup ===
# Logger konfigurieren, um in 'training.log' zu schreiben und auch auf der Konsole auszugeben
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"), # Loggt in Datei
        logging.StreamHandler(sys.stdout) # Loggt in Konsole
    ]
)
logger = logging.getLogger(__name__)


def get_git_hash():
    """ Ruft den aktuellen Git-Commit-Hash ab. """
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('ascii').strip()
        return git_hash
    except Exception as e:
        logger.warning(f"Konnte Git-Hash nicht abrufen: {e}. Setze auf 'N/A'.")
        return "N/A"

# NEUE HILFSFUNKTION FÜR R^2
def calculate_sse_total(labels):
    """ 
    Berechnet die Total Sum of Squares (SSE_Total) für R2.
    Dies ist die Baseline-Varianz, die erklärt werden soll.
    """
    
    # Sicherstellen, dass die Labels auf CPU sind für die Berechnung
    labels_cpu = labels.cpu()
    
    # 1. Mittelwert berechnen
    mean_label = torch.mean(labels_cpu)
    
    # 2. SSE_Total berechnen: Summe der quadrierten Abweichungen vom Mittelwert
    sse_total = torch.sum((labels_cpu - mean_label) ** 2)
    
    return sse_total.item() # Rückgabe als Standard-Python-Float

# %%
# ========================================================================
# HAUPT-SKRIPT-LOGIK (IN main() GEKAPSELT)
# ========================================================================

def main(args):
    
    # === 4.A.3: Start loggen ===
    logger.info("="*50)
    logger.info("STARTING NEW TRAINING RUN")
    logger.info(f"Git Commit Hash: {get_git_hash()}")
    logger.info(f"Verwendete Argumente: {args}")
    logger.info("="*50)

    # ###############################################################
    # SCHRITT 1: EXPLORATORY DATA ANALYSIS (EDA)
    # ###############################################################
    
    logger.info("--- START SCHRITT 1: EDA & DATEIPFADE ---")

    # DATEIPFADE
    # ============================================================
    notebook_dir = Path().resolve()
    data_csv = notebook_dir / "2025_10_13_mlops_biomass_data/mlops_biomass_data/digital_biomass_labels.xlsx"
    images_dir = notebook_dir / "2025_10_13_mlops_biomass_data/mlops_biomass_data/images_med_res"
    figures_dir = notebook_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Prüfen, ob Dateien und Ordner existieren
    logger.info(f"Existiert die Excel-Datei? {data_csv.exists()}")
    logger.info(f"Existiert der Bilderordner? {images_dir.exists()}")
    if not data_csv.exists() or not images_dir.exists():
        logger.error("Daten- oder Bildpfad nicht gefunden. Skript wird beendet.")
        return # Beendet die main-Funktion

    # METADATEN LADEN
    # ============================================================
    logger.info("Lade Metadaten...")
    df = pd.read_excel(data_csv)
    logger.info(f"Metadaten geladen. Spalten: {df.columns.tolist()}")


    # TARGET-DISTRIBUTION (fresh_weight_total)
    # ============================================================
    logger.info("Erstelle Plot: Target-Distribution")
    plt.figure(figsize=(8,5))
    plt.hist(df['fresh_weight_total'].dropna(), bins=50, color='green', alpha=0.7)
    plt.xlabel('fresh_weight_total (g)')
    plt.ylabel('Count')
    plt.title('Target Distribution - fresh_weight_total')
    plt.tight_layout()
    plt.savefig(figures_dir / 'target_distribution.png')
    plt.show()


    # STICH-PROBENHAFTE BILDER MIT LABELS
    # ============================================================
    logger.info("Erstelle Plot: Beispielbilder")
    unique_filenames = df['filename'].dropna().unique()
    if len(unique_filenames) >= 9:
        sample_filenames = random.sample(list(unique_filenames), 9)
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = axs.flatten()

        for i, fname in enumerate(sample_filenames):
            img_path = images_dir / fname
            if img_path.exists():
                img = Image.open(img_path)
            else:
                img = Image.new('RGB', (256,256), color=(200,200,200)) # Platzhalter
            
            axs[i].imshow(img)
            # Sicherstellen, dass der Wert existiert, bevor .values[0] aufgerufen wird
            weight_series = df[df['filename'] == fname]['fresh_weight_total']
            weight_val = weight_series.values[0] if not weight_series.empty else np.nan
            axs[i].set_title(f'{weight_val:.1f} g')
            axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(figures_dir / 'sample_images.png')
        plt.show()
    else:
        logger.warning("Nicht genügend eindeutige Bilder für den Sample-Plot gefunden.")


    # KORRELATIONSMATRIX DER NUMERISCHEN FEATURES (***FINAL KORRIGIERT***)
    # ============================================================
    logger.info("Erstelle Plot: Korrelations-Heatmap (Final Reduziert)")
    
    # *** FINALE KORREKTUR: ***
    # Nur EINEN Vertreter der Beleuchtungssensoren behalten,
    # um die '1en' (Multikollinearität) zu entfernen.
    numeric_cols = [
        'temperature', 
        'humidity', 
        'illuminancelux', # <--- NUR EINE Spalte für Beleuchtung
        'age_days', 
        'total_leaves', 
        'fresh_weight_total'
    ]
    
    # Sicherstellen, dass nur existierende Spalten verwendet werden
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if not valid_numeric_cols:
        logger.warning("Keine der numerischen Spalten für die Korrelationsmatrix gefunden.")
    else:
        corr = df[valid_numeric_cols].corr()

        # Figurengröße kann jetzt kleiner sein
        plt.figure(figsize=(8, 6)) 
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap (Final Reduced Features)')
        plt.tight_layout()
        # Dateiname geändert, um die neue Version zu speichern
        plt.savefig(figures_dir / 'correlation_heatmap.png') 
        plt.show()


    #  BEZIEHUNG: ALTER vs. BIOMASSE
    # ============================================================
    logger.info("Erstelle Plot: Alter vs. Biomasse")
    if 'age_days' in df.columns:
        plt.figure(figsize=(8,5))
        plt.scatter(df['age_days'], df['fresh_weight_total'], alpha=0.6, color='brown')
        plt.xlabel('age_days')
        plt.ylabel('fresh_weight_total (g)')
        plt.title('Age vs Biomass')
        plt.tight_layout()
        plt.savefig(figures_dir / 'age_vs_biomass.png')
        plt.show()
    else:
        logger.warning("Spalte 'age_days' nicht gefunden. Überspringe Alter vs Biomasse Plot.")


    # PIXELANALYSE DER BILDER (R/G/B MITTELWERTE)
    # ============================================================
    logger.info("Starte Pixelanalyse (kann dauern)...")
    means = []
    filenames_to_analyze = df['filename'].dropna()
    total = len(filenames_to_analyze)

    for i, fname in enumerate(filenames_to_analyze):
        img_path = images_dir / fname
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img) / 255.0
            means.append(arr.mean(axis=(0,1)))
        
        # Fortschritt inline anzeigen
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Berechne Bild {i+1}/{total}", end='\r')


    means = np.array(means)
    print("\nFertig!") # Zeilenumbruch nach dem Fortschrittsbalken
    logger.info("Pixelanalyse abgeschlossen.")

    if len(means) > 0:
        plt.figure(figsize=(8,5))
        plt.boxplot([means[:,0], means[:,1], means[:,2]], labels=['R','G','B'])
        plt.ylabel('Mean Pixel Value (normalized)')
        plt.title('Image Pixel Analysis')
        plt.tight_layout()
        plt.savefig(figures_dir / 'image_pixel_analysis.png')
        plt.show()
    else:
        logger.warning("Keine Bilder für Pixelanalyse gefunden.")


    # DATENQUALITÄTSPRÜFUNG
    # ============================================================
    logger.info("--- Datenqualitätsprüfung ---")
    missing_labels = df['fresh_weight_total'].isna().sum()
    logger.info(f'Fehlende Labels: {missing_labels}')

    # Sicherstellen, dass nur existierende Dateinamen geprüft werden
    missing_images = sum([not (images_dir / str(f)).exists() for f in df['filename'].dropna()])
    logger.info(f'Fehlende Bilddateien: {missing_images}')

    outliers = df['fresh_weight_total'].describe()
    logger.info(f'Statistik der Zielvariable:\n{outliers}')

    logger.info("--- ENDE SCHRITT 1: EDA ---")


    # %%
    # ###############################################################
    #
    # SCHRITT 2: DATA LOADING & PREPROCESSING
    #
    # ###############################################################

    logger.info("\n\n--- START SCHRITT 2: DATA LOADING & PREPROCESSING ---")

    # === 2.1 Metadaten filtern und bereinigen ===
    logger.info("\nSchritt 2.1: Bereinige Metadaten...")
    df_clean = df[['filename', 'fresh_weight_total']].copy()

    # 2. Entferne Zeilen mit fehlenden Labels (fresh_weight_total)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['fresh_weight_total'])
    logger.info(f"  {initial_rows - len(df_clean)} Zeilen wegen fehlendem Label entfernt.")

    # 3. Entferne Zeilen, bei denen die Bilddatei fehlt
    def check_image_exists(filename):
        if pd.isna(filename):
            return False
        return (images_dir / str(filename)).exists()

    df_clean['image_exists'] = df_clean['filename'].apply(check_image_exists)

    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['image_exists'] == True]
    logger.info(f"  {initial_rows - len(df_clean)} Zeilen wegen fehlender Bilddatei entfernt.")

    # Aufräumen
    df_clean = df_clean.drop(columns=['image_exists']).reset_index(drop=True)
    logger.info(f"-> Verbleibende Samples für Training/Validierung: {len(df_clean)}")
    
    if len(df_clean) == 0:
        logger.error("Keine Daten zum Trainieren übrig. Skript wird beendet.")
        return


    # === 2.2 Train/Validation Split (80/20) ===
    logger.info("\nSchritt 2.2: Erstelle Train/Validation Split (80/20)...")
    train_df, val_df = train_test_split(
        df_clean,
        test_size=0.2,       # 20% für die Validierung
        random_state=42      # Für reproduzierbare Ergebnisse
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(f"  Trainings-Samples: {len(train_df)}")
    logger.info(f"  Validierungs-Samples: {len(val_df)}")


    # === 2.3 Bild-Transformationen definieren ===
    logger.info("\nSchritt 2.3: Definiere Bild-Transformationen (Resize, Normalize)...")
    
    #  Transformationen für das Training (mit Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      # 1. Auf 224x224 skalieren
        transforms.RandomHorizontalFlip(),  # 2. Zufällig horizontal spiegeln
        transforms.RandomVerticalFlip(),    # 3. Zufällig vertikal spiegeln
        transforms.RandomRotation(20),      # 4. Zufällig um bis zu 20 Grad drehen
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 5. Farbton leicht ändern
        transforms.ToTensor(),              # 6. In PyTorch-Tensor umwandeln (0.0-1.0)
        transforms.Normalize(               # 7. Normalisieren (ImageNet-Statistiken)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) 
    ])
    
    # Transformationen für die Validierung (KEINE Augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      # 1. Auf 224x224 skalieren
        transforms.ToTensor(),              # 2. In PyTorch-Tensor umwandeln
        transforms.Normalize(               # 3. Normalisieren (ImageNet-Statistiken)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) 
    ])
    
    logger.info("  Transformationen für Training (mit Augmentation) und Validierung (ohne) sind definiert.")


    # === 2.4 Die Dataset-Klasse implementieren ===
    logger.info("\nSchritt 2.4: Definiere die PyTorch Dataset-Klasse...")
    class PlantBiomassDataset(Dataset):
        """Eigene Dataset-Klasse für die Biomasse-Daten."""
        
        def __init__(self, dataframe, image_dir, transform=None):
            self.df = dataframe
            self.image_dir = Path(image_dir)
            self.transform = transform

        def __len__(self):
            """Gibt die Gesamtanzahl der Samples zurück."""
            return len(self.df)

        def __getitem__(self, idx):
            """Lädt ein einzelnes Sample (Bild + Label) an der Position 'idx'."""
            
            # 1. Informationen aus dem DataFrame holen
            row = self.df.iloc[idx]
            img_filename = row['filename']
            label = row['fresh_weight_total']
            
            # 2. Bildpfad erstellen und Bild laden
            img_path = self.image_dir / img_filename
            image = Image.open(img_path).convert('RGB') # Immer in RGB konvertieren
            
            # 3. Transformationen auf das Bild anwenden
            if self.transform:
                image_tensor = self.transform(image)
            
            # 4. Label in einen Float-Tensor umwandeln (wichtig für Regression!)
            label_tensor = torch.tensor([label], dtype=torch.float32)
            
            return image_tensor, label_tensor

    logger.info("  PlantBiomassDataset Klasse ist definiert.")


    # === 2.5 DataLoader erstellen (***ANGEPASST***) ===
    logger.info("\nSchritt 2.5: Erstelle die finalen DataLoader...")

    # 1. Dataset-Objekte erstellen
    # ***KORREKTUR:*** Verwendet jetzt 'train_transforms'
    train_dataset = PlantBiomassDataset(
        dataframe=train_df,
        image_dir=images_dir,
        transform=train_transforms 
    )

    # ***KORREKTUR:*** Verwendet jetzt 'val_transforms'
    val_dataset = PlantBiomassDataset(
        dataframe=val_df,
        image_dir=images_dir,
        transform=val_transforms
    )

    # 2. Batch-Größe
    BATCH_SIZE = args.batch_size 

    # 3. DataLoader erstellen
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # WICHTIGER FIX: 0 statt 4
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0  # WICHTIGER FIX
    )

    logger.info(f"  train_loader und val_loader sind mit Batch-Größe {BATCH_SIZE} bereit.")


    # === 2.6 Testlauf: Funktioniert der Loader? ===
    logger.info("\nSchritt 2.6: Teste den train_loader...")

    try:
        # Hole einen einzelnen Batch
        images, labels = next(iter(train_loader))

        logger.info("\n---   Test-Batch erfolgreich geladen! ---")
        logger.info(f"Form der Bilder (Batch): {images.shape}")
        logger.info(f"Form der Labels (Batch): {labels.shape}")
        logger.info(f"Datentyp der Bilder:     {images.dtype}")
        logger.info(f"Datentyp der Labels:     {labels.dtype}")
        
    except Exception as e:
        logger.error(f"\n---   Fehler beim Laden des Test-Batch ---")
        logger.error(f"Fehler: {e}")
        return # Beenden, wenn Datenladen fehlschlägt

    logger.info("\n--- SCHRITT 2 (DATA LOADING) ABGESCHLOSSEN ---")


    # %%
    # ###############################################################
    # SCHRITT 3: MODEL ARCHITECTURE (ResNet Regression)
    # ###############################################################

    logger.info("\n--- START SCHRITT 3: MODEL ARCHITECTURE ---")

    # === 3.1: Device (GPU/CPU) wählen ===
# === 3.1: Device (GPU/CPU/MPS) wählen ===
    if torch.cuda.is_available():
        # Für NVIDIA GPUs
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Für Apple Silicon (M1/M2/M3) GPUs
        device = torch.device("mps")
    else:
        # Fallback auf CPU
        device = torch.device("cpu")

    logger.info(f"Rechengerät: {device}")

    # === 3.2: Pretrained ResNet laden ===
    logger.info(f"\nLade pretrained {args.model_name}...")
    
    if args.model_name == 'resnet18':
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 1) # Regression Head
    elif args.model_name == 'resnet50':
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 1) # Regression Head
    
    logger.info(f"{args.model_name} Basisnetz geladen. Feature-Dimension: {num_features}")
    logger.info("Letzter Layer ersetzt -> Linear(in_features, 1)")

    # === 3.4: Modell finalisieren ===
    model = base_model.to(device)
    model.train()   # Modell in Trainingsmodus setzen

    # === 3.5: Loss-Funktion & Optimizer ===
    criterion = nn.MSELoss()  # Mean Squared Error für Regression
    
    # Lernrate aus argparse
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    logger.info(f"Optimizer: Adam, Lernrate: {args.learning_rate}")

    # === 3.6: Zusammenfassung ===
    logger.info("\n---   Modell, Loss & Optimizer bereit ---")


    # === 3.7: Testlauf mit einem Batch ===
    try:
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        logger.info(f"\nTestausgabe Shape: {outputs.shape} (sollte [{BATCH_SIZE}, 1] oder weniger sein)")
    except Exception as e:
        logger.error(f"\n Fehler beim Modelltest: {e}")
        return

    
    # %%
    # ###############################################################
    #
    # SCHRITT 4: TRAINING LOOP (INKLUSIVE R^2)
    #
    # ###############################################################

    logger.info("\n--- START SCHRITT 4: TRAINING LOOP ---")

    # === 4.1: Setup für Training ===
    NUM_EPOCHS = args.epochs
    best_val_loss = float('inf')
    # R2: Wir maximieren R2, also starten wir mit einem niedrigen Wert
    best_val_r2 = float('-inf') 

    # Checkpoint-Ordner (zum Speichern des Modells während des Trainings)
    checkpoints_dir = notebook_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    # Pfad angepasst, um Modellnamen zu enthalten
    CHECKPOINT_PATH = checkpoints_dir / f"best_biomass_model_{args.model_name}.pth"

    # Results-Ordner
    results_dir = notebook_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    # =================================================

    # Listen, um den Loss-Verlauf für das Plotten zu speichern
    train_loss_history = []
    val_loss_history = []

    logger.info(f"Starte Training für {NUM_EPOCHS} Epochen...")
    logger.info(f"Modell wird auf '{device}' trainiert.")
    logger.info(f"Bestes Modell wird gespeichert unter: {CHECKPOINT_PATH}")
    logger.info(f"Ergebnisse (Plot, Metriken) werden gespeichert unter: {results_dir}")


    # NEUE BERECHNUNG: SSE_Total für den Validierungs-Satz (Baseline für R2)
    # ====================================================================
    # Wir müssen alle Labels einmal sammeln, um den Gesamtdurchschnitt zu berechnen.
    all_val_labels = torch.cat([labels for _, labels in val_loader]).to(device) 
    SSE_TOTAL = calculate_sse_total(all_val_labels) 
    
    # Stellen Sie sicher, dass SSE_TOTAL nicht Null ist
    if SSE_TOTAL <= 1e-6: # Prüfung auf sehr kleine Zahl statt exakt 0
        logger.error(f"SSE_Total ist fast Null ({SSE_TOTAL}). R2 kann nicht sinnvoll berechnet werden. Skript wird beendet.")
        return 

    logger.info(f"Basis für R2 (SSE_Total) auf Validierungs-Set berechnet: {SSE_TOTAL:.2f}")
    logger.info(f"Epoch training, bitte warten...")
    # ====================================================================


    # === 4.2: Der Haupt-Trainingsloop ===

    for epoch in range(NUM_EPOCHS):
        
        epoch_start_time = time.time()
        
        # --- Trainings-Phase ---
        model.train() 
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        # --- Validierungs-Phase ---
        model.eval()
        running_val_loss = 0.0
        
        # R2-Erweiterung: Sammeln der Residuen für SSE_Residual
        SSE_Residual = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels) # MSE für Batch
                
                # Berechnung des SSE_Residual für den Batch: Summe der quadrierten Fehler
                SSE_Residual += torch.sum((labels - outputs) ** 2).item()
                
                running_val_loss += loss.item() # Summieren des Loss-Werts

        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        epoch_val_rmse = math.sqrt(epoch_val_loss)

        # R2-Erweiterung: Berechnung des R2 für die Epoche
        # R2 = 1 - (SSE_Residual / SSE_Total)
        epoch_val_r2 = 1 - (SSE_Residual / SSE_TOTAL)

        epoch_duration = time.time() - epoch_start_time

        # --- Epoch-Zusammenfassung und Checkpoint ---
        logger.info(f"\n--- Epoche {epoch+1}/{NUM_EPOCHS} ---")
        logger.info(f"  Zeit: {epoch_duration:.2f}s")
        logger.info(f"  Train Loss (MSE): {epoch_train_loss:.4f}")
        logger.info(f"  Val Loss (MSE):   {epoch_val_loss:.4f}")
        logger.info(f"  Val RMSE:         {epoch_val_rmse:.2f} g")
        logger.info(f"  Val R-Squared (R2): {epoch_val_r2:.4f}") # R2 Metrik

        # Bestes Modell speichern (basierend auf Val Loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_r2 = epoch_val_r2 # R2 des besten Loss-Modells speichern
            try:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                logger.info(f" -> Neues bestes Modell gespeichert! (Val Loss: {best_val_loss:.4f}, R2: {best_val_r2:.4f})")
            except Exception as e:
                logger.error(f"FEHLER beim Speichern des Modells: {e}")

    logger.info("\n--- TRAINING ABGESCHLOSSEN ---")
    logger.info(f"Bestes Modell (Validierungs-Loss: {best_val_loss:.4f}, R2: {best_val_r2:.4f}) gespeichert unter {CHECKPOINT_PATH}")


    # === 4.3: Loss-Verlauf plotten ===
    logger.info("\nErstelle Plot für Loss-Verlauf...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Training Loss (MSE)')
        plt.plot(val_loss_history, label='Validation Loss (MSE)')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.title('Trainings- und Validierungs-Loss über die Zeit')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        loss_curve_path = results_dir / 'training_curves.png'
        plt.savefig(loss_curve_path)
        plt.show() # Behalte show() für interaktive Nutzung

        logger.info(f"Loss-Plot gespeichert unter {loss_curve_path}")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen/Speichern des Loss-Plots: {e}")


    # === 4.4: Metriken in metrics.txt speichern (mit R2) ===
    logger.info(f"Speichere finale Metriken in {results_dir / 'metrics.txt'}...")

    # Beste Metriken holen
    best_val_mse = best_val_loss
    best_val_rmse = math.sqrt(best_val_mse)

    try:
        with open(results_dir / 'metrics.txt', 'w') as f:
            f.write("Finale Validierungs-Metriken (bestes Modell):\n")
            f.write("=============================================\n")
            f.write(f"Best Validation MSE: {best_val_mse:.4f}\n")
            f.write(f"Best Validation RMSE: {best_val_rmse:.2f} g\n")
            f.write(f"Best Validation R-Squared (R2): {best_val_r2:.4f}\n") # R2 Metrik
            f.write("\nVerwendete Argumente:\n")
            f.write(str(args))
        logger.info("Metriken erfolgreich gespeichert.")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Metriken: {e}")

    logger.info("\n--- SCHRITT 4 (TRAINING & RESULTS) ABGESCHLOSSEN ---")


# %%
# ========================================================================
# SCRIPT-AUSFÜHRUNG
# ========================================================================

if __name__ == "__main__":
    # Dieser Block stellt sicher, dass der Code nur ausgeführt wird, 
    # wenn das Skript direkt gestartet wird (nicht beim Importieren).
    
    # 1. Argumente parsen
    # HINWEIS: In einer interaktiven Umgebung (wie Jupyter) 
    # args = parser.parse_args() mit leeren Argumenten füllen:
    
    # Für normale Skript-Ausführung:
    args = parser.parse_args()
    
    # Für interaktive Nutzung (z.B. Jupyter):
    # args = parser.parse_args(args=[]) 
    
    # 2. Hauptfunktion mit den Argumenten aufrufen
    main(args)