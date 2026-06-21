import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "src" / "training_loop.py"
EPOCH_PATTERN = re.compile(
    r"Ep\s+(\d+)/\d+.*Patient BalAcc\s+([0-9.]+).*Patient AUC\s+([0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimizacion reproducible de hiperparametros ResNet50 con Optuna."
    )
    parser.add_argument("--study-name", type=str, default="resnet50_tfm")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Numero total objetivo de trials. Repetir el comando reanuda hasta este total.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--roi-fracs", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-hours", type=float, default=None)

    parser.add_argument("--csv", type=str, default="data_index.csv")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D_Atlas")
    parser.add_argument("--output-root", type=str, default="models/optuna")
    parser.add_argument("--results-root", type=str, default="resultados/optuna")

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Registra tambien cada trial en Weights & Biases.",
    )
    parser.add_argument(
        "--pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detiene trials poco prometedores mediante MedianPruner.",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Conserva checkpoints de todos los trials, no solo el mejor.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def training_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_args(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.study_name):
        raise ValueError("study-name solo puede contener letras, numeros, guiones y guiones bajos.")
    if args.n_trials < 1:
        raise ValueError("n_trials debe ser mayor que cero.")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs y patience deben ser mayores que cero.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("Los batch sizes deben ser positivos.")
    if any(not 0 < roi_frac <= 1 for roi_frac in args.roi_fracs):
        raise ValueError("roi-fracs debe contener fracciones entre 0 y 1, por ejemplo 0.8.")


def sample_parameters(trial, args):
    return {
        "lr": trial.suggest_float("lr", 5e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 2e-3, log=True),
        "dropout": trial.suggest_categorical("dropout", [0.2, 0.3, 0.4, 0.5, 0.6]),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1, 0.15]),
        "balance_strategy": trial.suggest_categorical(
            "balance_strategy", ["sampler", "class_weights"]
        ),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        "roi_frac": trial.suggest_categorical("roi_frac", args.roi_fracs),
        "scheduler_factor": trial.suggest_categorical("scheduler_factor", [0.3, 0.5]),
    }


def build_trial_command(args, params, run_name, output_dir, excel_path):
    command = [
        sys.executable,
        "-u",
        str(TRAINING_SCRIPT),
        "--model", "resnet50",
        "--csv", args.csv,
        "--images", args.images,
        "--output-dir", training_path(output_dir),
        "--excel", training_path(excel_path),
        "--run-name", run_name,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--monitor", "val_patient_roc_auc",
        "--min-delta", "0.001",
        "--freeze", "layer4",
        "--optimizer", "adamw",
        "--scheduler", "plateau",
        "--scheduler-patience", "2",
        "--scheduler-factor", str(params["scheduler_factor"]),
        "--lr", str(params["lr"]),
        "--weight-decay", str(params["weight_decay"]),
        "--dropout", str(params["dropout"]),
        "--label-smoothing", str(params["label_smoothing"]),
        "--balance-strategy", params["balance_strategy"],
        "--batch-size", str(params["batch_size"]),
        "--roi-frac", str(params["roi_frac"]),
        "--seed", str(args.seed),
        "--wandb" if args.wandb else "--no-wandb",
    ]
    return command


def read_trial_metrics(excel_path, run_name):
    if not excel_path.exists():
        raise RuntimeError(f"No se genero el Excel esperado: {excel_path}")

    summary = pd.read_excel(excel_path, sheet_name="resumen")
    matches = summary[summary["run_name"].astype(str) == run_name]
    if matches.empty:
        raise RuntimeError(f"No se encontro el run {run_name} en {excel_path}")

    row = matches.iloc[-1]
    metric_names = [
        "best_epoch",
        "duracion_min",
        "best_val_patient_roc_auc",
        "best_val_patient_balanced_acc",
        "best_val_patient_specificity",
        "best_val_patient_sensitivity",
        "best_val_patient_mcc",
        "best_val_loss",
    ]
    metrics = {}
    for name in metric_names:
        if name in row.index and pd.notna(row[name]):
            metrics[name] = float(row[name])
    return metrics


def stop_process(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_trial_process(trial, command, log_path, pruning):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

                match = EPOCH_PATTERN.search(line)
                if not match:
                    continue

                epoch = int(match.group(1))
                patient_auc = float(match.group(3))
                trial.report(patient_auc, step=epoch)
                if pruning and trial.should_prune():
                    stop_process(process)
                    raise optuna.TrialPruned(
                        f"Trial podado en epoca {epoch} con patient AUC={patient_auc:.4f}"
                    )
    except KeyboardInterrupt:
        stop_process(process)
        raise

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"El entrenamiento termino con codigo {return_code}")


def make_objective(args, output_dir, excel_path, logs_dir):
    def objective(trial):
        params = sample_parameters(trial, args)
        run_name = f"optuna_{args.study_name}_trial_{trial.number:03d}"
        checkpoint = output_dir / f"{run_name}_best.pth"
        log_path = logs_dir / f"trial_{trial.number:03d}.log"

        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("checkpoint", str(checkpoint))
        trial.set_user_attr("log_path", str(log_path))

        command = build_trial_command(args, params, run_name, output_dir, excel_path)
        trial.set_user_attr("command", shlex.join(command))

        print("\n" + "=" * 100)
        print(f"OPTUNA TRIAL {trial.number}: {run_name}")
        print(json.dumps(params, indent=2, ensure_ascii=True))
        print("Comando:", shlex.join(command))
        print("=" * 100)

        if args.dry_run:
            return 0.5

        started = time.time()
        run_trial_process(trial, command, log_path, args.pruning)
        metrics = read_trial_metrics(excel_path, run_name)

        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("wall_time_minutes", (time.time() - started) / 60)

        objective_value = metrics.get("best_val_patient_roc_auc")
        if objective_value is None:
            raise RuntimeError("El trial no produjo best_val_patient_roc_auc.")
        return objective_value

    return objective


def cleanup_checkpoints(study, trial, output_dir, keep_all):
    if keep_all:
        return

    try:
        best_checkpoint = Path(study.best_trial.user_attrs["checkpoint"])
    except (ValueError, KeyError):
        best_checkpoint = None

    for stored_trial in study.trials:
        checkpoint_value = stored_trial.user_attrs.get("checkpoint")
        if not checkpoint_value:
            continue
        checkpoint = Path(checkpoint_value)
        if checkpoint == best_checkpoint:
            continue
        if checkpoint.parent == output_dir and checkpoint.exists():
            checkpoint.unlink()


def export_study_plots(study, args, results_dir):
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    generated = {}
    history_path = results_dir / f"{args.study_name}_optimization_history.png"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            axis = plot_optimization_history(study)
        axis.set_title("Optuna: evolución de ROC-AUC por paciente")
        axis.figure.tight_layout()
        axis.figure.savefig(history_path, dpi=200, bbox_inches="tight")
        plt.close(axis.figure)
        generated["optimization_history"] = str(history_path)
    except Exception as exc:
        print(f"[WARNING] No se pudo generar la grafica de historial: {exc}")

    if len(completed) >= 3:
        importance_path = results_dir / f"{args.study_name}_param_importances.png"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                axis = plot_param_importances(study)
            axis.set_title("Optuna: importancia de hiperparámetros")
            axis.figure.tight_layout()
            axis.figure.savefig(importance_path, dpi=200, bbox_inches="tight")
            plt.close(axis.figure)
            generated["param_importances"] = str(importance_path)
        except Exception as exc:
            print(f"[WARNING] No se pudo generar la grafica de importancias: {exc}")

    return generated


def export_study(study, args, results_dir):
    trials_csv = results_dir / f"{args.study_name}_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        print(f"No hay trials completos. Historial exportado en: {trials_csv}")
        return None

    best = study.best_trial
    plot_paths = export_study_plots(study, args, results_dir)
    best_data = {
        "study_name": args.study_name,
        "objective": "maximize val_patient_roc_auc",
        "best_trial": best.number,
        "best_value": best.value,
        "params": best.params,
        "validation_metrics": {
            key: value
            for key, value in best.user_attrs.items()
            if key.startswith("best_val_") or key in {"best_epoch", "duracion_min"}
        },
        "checkpoint": best.user_attrs.get("checkpoint"),
        "sqlite": str(results_dir / f"{args.study_name}.db"),
        "trials_csv": str(trials_csv),
        "plots": plot_paths,
    }

    best_json = results_dir / f"{args.study_name}_best_params.json"
    best_json.write_text(
        json.dumps(best_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_args = [
        "src/main.py",
        "--model", "resnet50",
        "--csv", args.csv,
        "--images", args.images,
        "--epochs", "60",
        "--patience", "8",
        "--monitor", "val_patient_roc_auc",
        "--freeze", "layer4",
        "--optimizer", "adamw",
        "--scheduler", "plateau",
        "--scheduler-patience", "3",
        "--scheduler-factor", str(best.params["scheduler_factor"]),
        "--lr", str(best.params["lr"]),
        "--weight-decay", str(best.params["weight_decay"]),
        "--dropout", str(best.params["dropout"]),
        "--label-smoothing", str(best.params["label_smoothing"]),
        "--balance-strategy", best.params["balance_strategy"],
        "--batch-size", str(best.params["batch_size"]),
        "--roi-frac", str(best.params["roi_frac"]),
        "--seed", str(args.seed),
        "--run-name", f"resnet50_optuna_{args.study_name}_final",
        "--eval-test",
        "--wandb" if args.wandb else "--no-wandb",
    ]
    commands = {
        "mac": shlex.join(["caffeinate", "python", *final_args]),
        "windows": subprocess.list2cmdline(["python", *final_args]),
    }
    commands_path = results_dir / f"{args.study_name}_best_commands.txt"
    commands_path.write_text(
        f"MAC\n{commands['mac']}\n\nWINDOWS\n{commands['windows']}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(f"MEJOR TRIAL: {best.number} | Patient ROC-AUC validacion: {best.value:.4f}")
    print(json.dumps(best.params, indent=2, ensure_ascii=False))
    print(f"SQLite: {best_data['sqlite']}")
    print(f"Trials CSV: {trials_csv}")
    print(f"Mejores parametros: {best_json}")
    print(f"Comandos finales: {commands_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"Grafica {plot_name}: {plot_path}")
    print("=" * 100)
    return best_data


def main():
    args = parse_args()
    validate_args(args)

    results_dir = project_path(args.results_root) / args.study_name
    output_dir = project_path(args.output_root) / args.study_name
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    excel_path = results_dir / f"{args.study_name}_training.xlsx"
    storage_path = results_dir / f"{args.study_name}.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        if args.pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    finished_trials = sum(trial.state in finished_states for trial in study.trials)
    remaining_trials = max(0, args.n_trials - finished_trials)

    print("\n" + "=" * 100)
    print(f"ESTUDIO OPTUNA: {args.study_name}")
    print(f"Objetivo: maximizar val_patient_roc_auc | Modelo: ResNet50 | Dispositivo: auto/MPS")
    print(f"Trials terminados: {finished_trials}/{args.n_trials} | Pendientes: {remaining_trials}")
    print(f"Persistencia: {storage_path}")
    print("El conjunto de test NO se utiliza durante esta busqueda.")
    print("=" * 100)

    if remaining_trials > 0:
        objective = make_objective(args, output_dir, excel_path, logs_dir)
        timeout = args.timeout_hours * 3600 if args.timeout_hours else None
        try:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout,
                n_jobs=1,
                gc_after_trial=True,
                callbacks=[
                    lambda current_study, current_trial: cleanup_checkpoints(
                        current_study,
                        current_trial,
                        output_dir,
                        args.keep_all_checkpoints,
                    )
                ],
                catch=(RuntimeError,),
            )
        except KeyboardInterrupt:
            print("\nBusqueda interrumpida por el usuario. El estudio queda guardado y puede reanudarse.")
    else:
        print("El estudio ya alcanzo el numero total solicitado de trials.")

    export_study(study, args, results_dir)


if __name__ == "__main__":
    main()
