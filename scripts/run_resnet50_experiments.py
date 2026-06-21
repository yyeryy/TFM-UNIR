import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "src" / "training_loop.py"
XAI_SCRIPT = PROJECT_ROOT / "src" / "explicabilidad.py"


EXPERIMENTS = [
    {
        "name": "roi80_lr1e5_reg_fuerte",
        "roi_frac": 80,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "label_smoothing": 0.1,
        "balance_strategy": "sampler",
    },
    {
        "name": "roi80_lr2e5_reg_fuerte",
        "roi_frac": 80,
        "lr": 2e-5,
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "label_smoothing": 0.1,
        "balance_strategy": "sampler",
    },
    {
        "name": "roi80_dropout03",
        "roi_frac": 80,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "balance_strategy": "sampler",
    },
    {
        "name": "roi80_smoothing005",
        "roi_frac": 80,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "label_smoothing": 0.05,
        "balance_strategy": "sampler",
    },
    {
        "name": "roi80_class_weights",
        "roi_frac": 80,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "label_smoothing": 0.1,
        "balance_strategy": "class_weights",
    },
    {
        "name": "roi70_lr1e5_reg_fuerte",
        "roi_frac": 70,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "label_smoothing": 0.1,
        "balance_strategy": "sampler",
    },
]

QUICK_EXPERIMENTS = {
    "roi80_lr1e5_reg_fuerte",
    "roi80_lr2e5_reg_fuerte",
    "roi80_dropout03",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta una busqueda secuencial y reproducible con ResNet50."
    )
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[experiment["name"] for experiment in EXPERIMENTS],
        help="Ejecuta solamente los experimentos indicados.",
    )
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--campaign-name", type=str, default=None)

    parser.add_argument("--csv", type=str, default="data_index.csv")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D_Atlas")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--excel", type=str, default="resultados/experimentos_entrenamiento.xlsx")
    parser.add_argument("--results-csv", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-xai-images", type=int, default=8)

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finalize-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evalua en test solo el mejor experimento de validacion.",
    )
    parser.add_argument(
        "--xai-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Genera Grad-CAM solamente para el mejor experimento.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_experiments(args):
    if args.only:
        selected_names = set(args.only)
    elif args.preset == "quick":
        selected_names = QUICK_EXPERIMENTS
    else:
        selected_names = {experiment["name"] for experiment in EXPERIMENTS}
    return [experiment.copy() for experiment in EXPERIMENTS if experiment["name"] in selected_names]


def print_experiments(experiments):
    print("\nExperimentos seleccionados:")
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"  {index}. {experiment['name']} | ROI={experiment['roi_frac']} | "
            f"lr={experiment['lr']:.1e} | wd={experiment['weight_decay']:.1e} | "
            f"dropout={experiment['dropout']} | smoothing={experiment['label_smoothing']} | "
            f"balance={experiment['balance_strategy']}"
        )


def build_training_command(args, experiment, run_name):
    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--model", "resnet50",
        "--csv", args.csv,
        "--images", args.images,
        "--output-dir", args.output_dir,
        "--excel", args.excel,
        "--run-name", run_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--scheduler", "plateau",
        "--scheduler-patience", str(args.scheduler_patience),
        "--scheduler-factor", str(args.scheduler_factor),
        "--min-delta", str(args.min_delta),
        "--monitor", "val_patient_balanced_acc",
        "--freeze", "layer4",
        "--roi-frac", str(experiment["roi_frac"]),
        "--lr", str(experiment["lr"]),
        "--weight-decay", str(experiment["weight_decay"]),
        "--dropout", str(experiment["dropout"]),
        "--label-smoothing", str(experiment["label_smoothing"]),
        "--balance-strategy", experiment["balance_strategy"],
        "--seed", str(args.seed),
    ]
    command.append("--wandb" if args.wandb else "--no-wandb")
    return command


def read_validation_metrics(excel_path, run_name):
    if not excel_path.exists():
        return {}

    try:
        summary = pd.read_excel(excel_path, sheet_name="resumen")
    except Exception as exc:
        print(f"[WARNING] No se pudo leer el Excel para {run_name}: {exc}")
        return {}
    if "run_name" not in summary.columns:
        return {}

    matches = summary[summary["run_name"].astype(str) == run_name]
    if matches.empty:
        return {}

    row = matches.iloc[-1]
    metric_columns = [
        "best_epoch",
        "duracion_min",
        "best_val_patient_balanced_acc",
        "best_val_patient_roc_auc",
        "best_val_patient_specificity",
        "best_val_patient_sensitivity",
        "best_val_patient_mcc",
        "best_val_loss",
    ]
    metrics = {}
    for column in metric_columns:
        if column in row.index and pd.notna(row[column]):
            value = row[column]
            metrics[column] = value.item() if hasattr(value, "item") else value
    return metrics


def save_results(records, results_path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(results_path, index=False)


def choose_best(records):
    completed = [
        record for record in records
        if record.get("status") in {"completed", "skipped"}
        and record.get("best_val_patient_balanced_acc") is not None
    ]
    if not completed:
        return None

    return max(
        completed,
        key=lambda record: (
            float(record.get("best_val_patient_balanced_acc", float("-inf"))),
            float(record.get("best_val_patient_roc_auc", float("-inf"))),
            float(record.get("best_val_patient_specificity", float("-inf"))),
        ),
    )


def print_ranking(records):
    ranking = pd.DataFrame(records)
    metric = "best_val_patient_balanced_acc"
    if ranking.empty or metric not in ranking.columns:
        return

    ranking[metric] = pd.to_numeric(ranking[metric], errors="coerce")
    ranking = ranking.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranking.empty:
        return

    columns = [
        "run_name",
        "best_val_patient_balanced_acc",
        "best_val_patient_roc_auc",
        "best_val_patient_specificity",
        "best_val_patient_sensitivity",
        "best_epoch",
    ]
    columns = [column for column in columns if column in ranking.columns]
    print("\nRANKING DE VALIDACION")
    print(ranking[columns].to_string(index=False))


def evaluate_best(args, best_record):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.evaluacion import evaluar_modelo_test

    checkpoint = Path(best_record["checkpoint"])
    metrics, _ = evaluar_modelo_test(
        checkpoint_path=checkpoint,
        csv_path=args.csv,
        images_dir=args.images,
    )
    for key, value in metrics.items():
        if isinstance(value, (int, float)) or value is None:
            best_record[f"final_test_{key}"] = value


def generate_best_xai(args, best_record):
    checkpoint = Path(best_record["checkpoint"])
    output_dir = Path("XAI") / best_record["run_name"]
    command = [
        sys.executable,
        str(XAI_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--csv", args.csv,
        "--images", args.images,
        "--output-dir", str(output_dir),
        "--num-images", str(args.num_xai_images),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    experiments = selected_experiments(args)
    print_experiments(experiments)

    if args.list_experiments:
        return

    campaign_name = args.campaign_name or datetime.now().strftime("r50_search_%Y%m%d_%H%M%S")
    results_path = resolve_project_path(
        args.results_csv or f"resultados/{campaign_name}_summary.csv"
    )
    excel_path = resolve_project_path(args.excel)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCampana: {campaign_name}")
    print(f"Resumen CSV: {results_path}")
    print("El test no se usa durante la busqueda; solo se evalua el ganador final.\n")

    records = []
    for index, experiment in enumerate(experiments, start=1):
        run_name = f"{campaign_name}_{experiment['name']}"
        checkpoint = output_dir / f"{run_name}_best.pth"
        record = {
            "campaign": campaign_name,
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "status": "pending",
            **experiment,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        }
        records.append(record)

        existing_metrics = read_validation_metrics(excel_path, run_name)
        if args.skip_existing and checkpoint.exists() and existing_metrics:
            print(f"[{index}/{len(experiments)}] OMITIDO: {run_name} ya esta completado.")
            record.update(existing_metrics)
            record["status"] = "skipped"
            save_results(records, results_path)
            continue

        command = build_training_command(args, experiment, run_name)
        print("\n" + "=" * 100)
        print(f"[{index}/{len(experiments)}] INICIANDO: {run_name}")
        print("Comando:", " ".join(command))
        print("=" * 100)

        if args.dry_run:
            record["status"] = "dry_run"
            continue

        started = time.time()
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            record["status"] = "completed"
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["return_code"] = exc.returncode
            print(f"[ERROR] {run_name} termino con codigo {exc.returncode}. Se continua con el siguiente.")
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["elapsed_minutes"] = (time.time() - started) / 60
            save_results(records, results_path)
            print("\nBusqueda interrumpida. El resumen parcial ha quedado guardado.")
            return

        record["elapsed_minutes"] = (time.time() - started) / 60
        record.update(read_validation_metrics(excel_path, run_name))
        save_results(records, results_path)

    save_results(records, results_path)
    if args.dry_run:
        print("\nDry run completado: no se ha entrenado ningun modelo.")
        return

    best = choose_best(records)
    if best is None:
        print("\nNo hay experimentos completados con metricas validas.")
        return

    print_ranking(records)
    print("\n" + "=" * 100)
    print("MEJOR CONFIGURACION DE VALIDACION")
    print(json.dumps(best, indent=2, ensure_ascii=True, default=str))
    print("=" * 100)

    if args.finalize_best:
        print("\nEvaluando una unica vez el ganador sobre el conjunto de test...")
        try:
            evaluate_best(args, best)
        except Exception as exc:
            best["final_test_error"] = str(exc)
            print(f"[ERROR] No se pudo evaluar el ganador en test: {exc}")
        save_results(records, results_path)

    if args.xai_best:
        print("\nGenerando XAI para el ganador...")
        try:
            generate_best_xai(args, best)
        except Exception as exc:
            best["xai_error"] = str(exc)
            save_results(records, results_path)
            print(f"[ERROR] No se pudo generar XAI para el ganador: {exc}")

    print(f"\nBusqueda terminada. Resumen: {results_path}")


if __name__ == "__main__":
    main()
