# MLOps Uni-Projekt: Biomasse-Vorhersage von Pflanzen

Dieses Repository wurde im Rahmen eines Uni-Projekts erstellt. Es demonstriert den Aufbau einer vollständigen Machine-Learning-Pipeline – von der ersten Datenanalyse bis zur automatisierten Orchestrierung und dem Experiment-Tracking.

##  Projektziel
Ziel ist die Vorhersage des Gesamtgewichts (`fresh_weight_total`) von Pflanzen basierend auf Bilddaten. Das Projekt ist in zwei aufeinander aufbauende Phasen unterteilt.

---

##  Teil 1: Datenanalyse & Modell-Basis (MLOPS-A1)
In der ersten Phase wurde das Fundament für das Training gelegt.

* **Explorative Datenanalyse (EDA):** Untersuchung der Datenqualität, wobei fehlende Labels und Bilder als Hauptproblem identifiziert wurden.
* **Erkenntnisse:** Die Zielvariable ist stark rechtsschief verteilt (viele leichte Pflanzen, wenige schwere). Das Alter der Pflanze korreliert am stärksten mit dem Gewicht.
* **Modellierung:** Implementierung eines **ResNet18** (Transfer Learning), das für eine Regressionsaufgabe angepasst wurde.
* **Training:** Erster Funktionstest mit PyTorch, MSE-Loss und Adam-Optimizer.

##  Teil 2: Pipeline-Automatisierung & MLOps (MLOPS-A2)
In der zweiten Phase wurde der manuelle Prozess in eine professionelle MLOps-Umgebung überführt.

* **Orchestrierung mit Dagster:** Der gesamte Workflow wurde in Assets unterteilt (`raw_dataset`, `preprocessed_data`, `trained_model`, `model_evaluation`), um die Reproduzierbarkeit sicherzustellen.
* **Experiment-Tracking mit MLflow:** * Automatisches Logging von Hyperparametern (Learning Rate, Batch Size, Epochen).
    * Tracking von Metriken wie Train/Val Loss und $R^2$.
    * Speicherung des "Best Models" als Artefakt direkt in MLflow.
* **Automatisierte Evaluation:** Erstellung von Scatter-Plots (Predicted vs. Actual), um die Modellgüte direkt nach dem Training zu bewerten.

---

##  Technologie-Stack
* **Frameworks:** PyTorch, Torchvision
* **MLOps:** Dagster (Orchestration), MLflow (Tracking)
* **Analyse:** Pandas, Matplotlib, Seaborn
* **Modell:** ResNet18 (CNN)

##  Installation & Ausführung
1. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   mlflow server --port 5001
   dagster dev -f MLOPS-A2/dagster_pipeline.py
   
