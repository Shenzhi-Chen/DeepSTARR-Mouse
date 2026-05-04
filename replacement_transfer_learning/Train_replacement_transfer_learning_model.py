#!/groups/stark/shenzhi.chen/.conda/envs/deepSTARR/bin/python

#########
### Load arguments
#########
import sys, getopt, os

def main(argv):
    fold = ""
    output_var = ""
    architecture = ""
    model_ID_output = ""
    fasta_path = ""
    label_path = ""
    vista_testset_fa = ""
    vista_testset_label = ""
    n_train_meta = None  # from meta

    try:
        opts, args = getopt.getopt(argv, "hi:v:a:o:q:p:t:u:s:")
    except getopt.GetoptError:
        print(
            "Train_transfer_learning_model.py "
            "-i <fold> -v <output variable> -a <architecture> -o <output model prefix> "
            "-p <fasta path> -q <label path> "
            "-t <vista_testset_fa> -u <vista_testset_label> "
            "-s <n_train>"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "Train_transfer_learning_model.py "
                "-i <fold> -v <output variable> -a <architecture> -o <output model prefix> "
                "-p <fasta path> -q <label path> "
                "-t <vista_testset_fa> -u <vista_testset_label> "
                "-s <n_train>"
            )
            sys.exit()
        elif opt == "-i":
            fold = arg
        elif opt == "-v":
            output_var = arg
        elif opt == "-a":
            architecture = arg
        elif opt == "-o":
            model_ID_output = arg
        elif opt == "-p":
            fasta_path = arg
        elif opt == "-q":
            label_path = arg
        elif opt == "-t":
            vista_testset_fa = arg
        elif opt == "-u":
            vista_testset_label = arg
        elif opt == "-s":
            try:
                n_train_meta = int(arg)
            except ValueError:
                sys.exit(f"Invalid -s n_train: {arg}")

    # required checks
    if fold == "":
        sys.exit("fold not found (-i)")
    if output_var == "":
        sys.exit("variable output not found (-v)")
    if architecture == "":
        sys.exit("architecture not found (-a)")
    if model_ID_output == "":
        sys.exit("Output model ID not found (-o)")
    if fasta_path == "":
        sys.exit("fasta path not found (-p)")
    if label_path == "":
        sys.exit("label path not found (-q)")
    if vista_testset_fa == "":
        sys.exit("vista test fasta not found (-t)")
    if vista_testset_label == "":
        sys.exit("vista test label file not found (-u)")
    if n_train_meta is None:
        sys.exit("n_train not found (-s)")

    if not os.path.exists(vista_testset_fa):
        sys.exit(f"vista test fasta does not exist: {vista_testset_fa}")
    if not os.path.exists(vista_testset_label):
        sys.exit(f"vista test label file does not exist: {vista_testset_label}")

    print("fold", fold)
    print("variable output", output_var)
    print("Model architecture", architecture)
    print("Output model prefix", model_ID_output)
    print("fasta path", fasta_path)
    print("label path", label_path)
    print("vista_testset_fa", vista_testset_fa)
    print("vista_testset_label", vista_testset_label)
    print("n_train_meta", n_train_meta)

    return fold, output_var, architecture, model_ID_output, fasta_path, label_path, vista_testset_fa, vista_testset_label, n_train_meta

if __name__ == "__main__":
    fold, output_var, architecture, model_ID_output, fasta_path, label_path, vista_testset_fa, vista_testset_label, n_train_meta = main(sys.argv[1:])


#########
### Libraries
#########
import random
random.seed(1234)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History

from sklearn.metrics import roc_auc_score, average_precision_score

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# helper for fasta
sys.path.append('/groups/stark/heinzl/Projects/DeepSTARR/DeepSTARR/Neural_Network_DNA_Demo/')
from helper import IOHelper, SequenceHelper

# wandb
import wandb
from wandb.integration.keras import WandbCallback

print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')), "\n")

#########
### Output dirs
#########
OUTPUT_DIR = os.path.dirname(model_ID_output) if os.path.dirname(model_ID_output) else "."
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval_plot")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

#########
### W&B init
#########
wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "shenzhichen"),
    project=os.environ.get("WANDB_PROJECT", "DeepSTARR_TF_revision"),
    config={
        "fold": fold,
        "output_var": output_var,
        "architecture": architecture,
        "model_ID_output": model_ID_output,
        "fasta_path": fasta_path,
        "label_path": label_path,
        "vista_testset_fa": vista_testset_fa,
        "vista_testset_label": vista_testset_label,
        "vista_label_col": output_var,
        "n_train_meta": int(n_train_meta),
        "seed": 1234,
    }
)

#########
### Helpers
#########
def prepare_input(fasta_path, label_path, fold, set_name, output_var):
    file_seq = f"{fasta_path}/{fold}_sequences_{set_name}.fa"
    if not os.path.exists(file_seq):
        raise FileNotFoundError(file_seq)

    input_fasta_data = IOHelper.get_fastas_from_file(file_seq, uppercase=True)
    if input_fasta_data.shape[0] == 0:
        raise ValueError(f"No sequences loaded from {file_seq}")

    seq_len = len(input_fasta_data.sequence.iloc[0])
    seq_matrix = SequenceHelper.do_one_hot_encoding(
        input_fasta_data.sequence,
        seq_len,
        SequenceHelper.parse_alpha_to_seq
    )
    X = np.nan_to_num(seq_matrix).astype(np.float32)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    activity_path = f"{label_path}/{fold}_sequences_activity_{set_name}.txt"
    if not os.path.exists(activity_path):
        raise FileNotFoundError(activity_path)

    Activity = pd.read_table(activity_path)
    if output_var not in Activity.columns:
        raise KeyError(f"Column '{output_var}' not found in {activity_path}")

    Y = Activity[output_var]
    Y = pd.get_dummies(Y)["active"].astype(np.int64)

    print(f"\nLoaded {set_name}: X {X.shape}, pos rate {Y.mean():.4f} (n_pos={int(Y.sum())}/{len(Y)})")
    return input_fasta_data, X, Y

def onehot_from_fasta(fa_path):
    if not os.path.exists(fa_path):
        raise FileNotFoundError(fa_path)
    fasta_df = IOHelper.get_fastas_from_file(fa_path, uppercase=True)
    if fasta_df.shape[0] == 0:
        raise ValueError(f"No sequences loaded from {fa_path}")

    seq_len = len(fasta_df.sequence.iloc[0])
    seq_matrix = SequenceHelper.do_one_hot_encoding(
        fasta_df.sequence,
        seq_len,
        SequenceHelper.parse_alpha_to_seq
    )
    X = np.nan_to_num(seq_matrix).astype(np.float32)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    return fasta_df, X

def read_label_table(path):
    # try TSV then CSV
    try:
        df = pd.read_table(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df

def to_binary_labels(series):
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        return (s.astype(float) != 0).astype(np.int64).values

    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "active": 1, "inactive": 0,
        "positive": 1, "negative": 0,
        "pos": 1, "neg": 0,
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
    }
    y = s.map(mapping)
    if y.isna().any():
        bad = s[y.isna()].unique()[:10]
        raise ValueError(f"Unrecognized label values (showing up to 10): {bad}")
    return y.astype(np.int64).values

def ppv_curve_and_max_with_min_tp(y_true, y_score, min_tp=100):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Length mismatch: y_true {y_true.shape}, y_score {y_score.shape}")

    order = np.argsort(-y_score)
    score_sorted = y_score[order]
    y_sorted = y_true[order].astype(np.int64)

    k = np.arange(1, len(y_sorted) + 1, dtype=np.int64)
    tp_cum = np.cumsum(y_sorted)
    ppv = tp_cum / k
    eligible = tp_cum >= int(min_tp)

    if not np.any(eligible):
        return {
            "order": order,
            "score_sorted": score_sorted,
            "y_sorted": y_sorted,
            "k": k,
            "tp_cum": tp_cum,
            "ppv": ppv,
            "eligible": eligible,
            "best_ppv": float("nan"),
            "best_k": None,
            "best_threshold": None,
        }

    eligible_idx = np.where(eligible)[0]
    best_idx = eligible_idx[np.argmax(ppv[eligible])]
    best_k = int(best_idx + 1)
    best_ppv = float(ppv[best_idx])
    best_threshold = float(score_sorted[best_idx])

    return {
        "order": order,
        "score_sorted": score_sorted,
        "y_sorted": y_sorted,
        "k": k,
        "tp_cum": tp_cum,
        "ppv": ppv,
        "eligible": eligible,
        "best_ppv": best_ppv,
        "best_k": best_k,
        "best_threshold": best_threshold,
    }

def save_ppv_curve_plot_and_tsv(out_dict, out_prefix):
    df = pd.DataFrame({
        "k": out_dict["k"],
        "ppv": out_dict["ppv"],
        "tp_cum": out_dict["tp_cum"],
        "eligible": out_dict["eligible"].astype(int),
        "score_sorted": out_dict["score_sorted"],
        "y_sorted": out_dict["y_sorted"],
    })
    tsv_path = out_prefix + ".tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    plt.figure()
    plt.plot(out_dict["k"], out_dict["ppv"])
    plt.xlabel("k (top-k predicted positives)")
    plt.ylabel("PPV(k) = TP(k)/k")
    plt.title("PPV curve (sorted by prediction score)")
    if out_dict["best_k"] is not None:
        bk = out_dict["best_k"]
        bp = out_dict["best_ppv"]
        plt.axvline(bk, linestyle="--")
        plt.scatter([bk], [bp])
        plt.text(bk, bp, f" best_k={bk}, maxPPV={bp:.3f}", fontsize=8)

    png_path = out_prefix + ".png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return tsv_path, png_path

#########
### Load datasets
#########
print("\nLoad fold sequences\n")
train_fasta, X_train, Y_train = prepare_input(fasta_path, label_path, fold, "training", output_var)
valid_fasta, X_valid, Y_valid = prepare_input(fasta_path, label_path, fold, "validation", output_var)
test_fasta,  X_test,  Y_test  = prepare_input(fasta_path, label_path, fold, "test", output_var)

# sanity check meta vs actual
n_train_actual = int(X_train.shape[0])
if n_train_actual != int(n_train_meta):
    print(f"[WARN] n_train_meta (-s)={n_train_meta}, but actual X_train.shape[0]={n_train_actual}. Using meta for LR selection.")

# 2-tier LR
if int(n_train_meta) < 10_000:
    lr = 3e-5
else:
    lr = 1e-4
print(f"[INFO] Using lr={lr} (n_train_meta={n_train_meta})")

print("\nLoad VISTA test fasta + labels\n")
vista_fasta_df, X_vista = onehot_from_fasta(vista_testset_fa)
vista_label_df = read_label_table(vista_testset_label)

label_col = output_var
if label_col not in vista_label_df.columns:
    raise KeyError(f"VISTA label column '{label_col}' not found in {vista_testset_label}. Available: {list(vista_label_df.columns)}")

Y_vista = to_binary_labels(vista_label_df[label_col])
if len(Y_vista) != X_vista.shape[0]:
    raise ValueError(f"VISTA label length {len(Y_vista)} != VISTA fasta sequences {X_vista.shape[0]}. Ensure label rows align with fasta order.")

print(f"Loaded VISTA: X {X_vista.shape}, pos rate {Y_vista.mean():.4f} (n_pos={int(Y_vista.sum())}/{len(Y_vista)})")

#########
### Load base model and build TL head
#########
def load_model_from_json_h5(prefix_path):
    weights_path = prefix_path + ".h5"
    json_path = prefix_path + ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing json: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing h5: {weights_path}")

    from tensorflow.keras.models import model_from_json
    model = model_from_json(open(json_path).read())
    model.load_weights(weights_path)
    model.summary()
    return model

print("\nInit model:", architecture, "\n")
base_model = load_model_from_json_h5(architecture)
DeepSTARR_bottleneck = Model(base_model.input, base_model.layers[-2].output)

params = {
    "batch_size": 128,
    "epochs": 200,
    "early_stop": 10,
    "lr": lr,
    "min_tp_for_max_ppv": 100,
}
wandb.config.update(params, allow_val_change=True)

inp = Input(shape=(1001, 4))
x = DeepSTARR_bottleneck(inp)
out = Dense(1, name="Activity")(x)
out = Activation("sigmoid")(out)

DeepSTARR_tl = Model(inp, out)
DeepSTARR_tl.compile(
    optimizer=Adam(learning_rate=params["lr"]),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(curve="ROC", name="AUROC"),
        tf.keras.metrics.AUC(curve="PR", name="AUPRC"),
        tf.keras.metrics.Precision(thresholds=0.5, name="Prec@0.5"),
        tf.keras.metrics.Recall(thresholds=0.5, name="Rec@0.5"),
    ],
)
DeepSTARR_tl.summary()

#########
### Train
#########
print("\nModel training\n")
callbacks = [
    EarlyStopping(patience=params["early_stop"], monitor="val_loss", restore_best_weights=True),
    History(),
    WandbCallback(save_model=False),
]

_ = DeepSTARR_tl.fit(
    X_train, Y_train,
    validation_data=(X_valid, Y_valid),
    batch_size=params["batch_size"],
    epochs=params["epochs"],
    callbacks=callbacks,
    verbose=1
)

#########
### Save model
#########
print("\nSaving model ...\n")
model_json = DeepSTARR_tl.to_json()
with open(model_ID_output + ".json", "w") as json_file:
    json_file.write(model_json)
DeepSTARR_tl.save_weights(model_ID_output + ".h5")

artifact = wandb.Artifact(name=os.path.basename(model_ID_output), type="keras_model")
artifact.add_file(model_ID_output + ".json")
artifact.add_file(model_ID_output + ".h5")
wandb.log_artifact(artifact)

#########
### Predict fold test set
#########
print("\nPredicting fold test set and saving outputs...\n")
pred_test = DeepSTARR_tl.predict(X_test, batch_size=params["batch_size"], verbose=0).reshape(-1)

test_pred_df = test_fasta.copy()
test_pred_df["Predictions"] = pred_test
if "sequence" in test_pred_df.columns:
    test_pred_df = test_pred_df.drop(columns=["sequence"])

pred_out_path = os.path.join(EVAL_DIR, "test_predictions.tsv")
test_pred_df.to_csv(pred_out_path, sep="\t", index=False)
print("Saved fold test predictions:", pred_out_path)

pred_art = wandb.Artifact(name=f"{os.path.basename(model_ID_output)}_test_predictions", type="predictions")
pred_art.add_file(pred_out_path)
wandb.log_artifact(pred_art)

#########
### Predict VISTA test
#########
print("\nPredicting VISTA test and saving outputs...\n")
pred_vista = DeepSTARR_tl.predict(X_vista, batch_size=params["batch_size"], verbose=0).reshape(-1)

vista_pred_df = vista_fasta_df.copy()
vista_pred_df["Predictions"] = pred_vista
if "sequence" in vista_pred_df.columns:
    vista_pred_df = vista_pred_df.drop(columns=["sequence"])

vista_pred_out_path = os.path.join(EVAL_DIR, "vista_test_predictions.tsv")
vista_pred_df.to_csv(vista_pred_out_path, sep="\t", index=False)
print("Saved VISTA test predictions:", vista_pred_out_path)

vista_pred_art = wandb.Artifact(name=f"{os.path.basename(model_ID_output)}_vista_test_predictions", type="predictions")
vista_pred_art.add_file(vista_pred_out_path)
wandb.log_artifact(vista_pred_art)

#########
### Evaluate fold test (AUROC/AUPRC + PPV)
#########
print("\nEvaluating on fold test set ...\n")
y_test_np = np.asarray(Y_test).reshape(-1)

fold_test_auroc = roc_auc_score(y_test_np, pred_test)
fold_test_auprc = average_precision_score(y_test_np, pred_test)
fold_test_ppv = ppv_curve_and_max_with_min_tp(y_test_np, pred_test, min_tp=params["min_tp_for_max_ppv"])

fold_ppv_prefix = os.path.join(EVAL_DIR, "ppv_curve_fold_test")
fold_ppv_tsv, fold_ppv_png = save_ppv_curve_plot_and_tsv(fold_test_ppv, fold_ppv_prefix)

wandb.log({
    "fold_test/AUROC": fold_test_auroc,
    "fold_test/AUPRC": fold_test_auprc,
    f"fold_test/max_ppv_minTP{params['min_tp_for_max_ppv']}": fold_test_ppv["best_ppv"],
    f"fold_test/best_k_minTP{params['min_tp_for_max_ppv']}": (fold_test_ppv["best_k"] if fold_test_ppv["best_k"] is not None else -1),
    f"fold_test/best_threshold_minTP{params['min_tp_for_max_ppv']}": (fold_test_ppv["best_threshold"] if fold_test_ppv["best_threshold"] is not None else np.nan),
    "fold_test/ppv_curve_plot": wandb.Image(fold_ppv_png),
})

fold_eval_art = wandb.Artifact(name=f"{os.path.basename(model_ID_output)}_fold_test_eval", type="eval")
fold_eval_art.add_file(fold_ppv_tsv)
fold_eval_art.add_file(fold_ppv_png)
wandb.log_artifact(fold_eval_art)

#########
### Evaluate VISTA test (AUROC/AUPRC + PPV)
#########
print("\nEvaluating on VISTA test set ...\n")
vista_auroc = roc_auc_score(Y_vista, pred_vista)
vista_auprc = average_precision_score(Y_vista, pred_vista)
vista_ppv = ppv_curve_and_max_with_min_tp(Y_vista, pred_vista, min_tp=params["min_tp_for_max_ppv"])

vista_ppv_prefix = os.path.join(EVAL_DIR, "ppv_curve_vista")
vista_ppv_tsv, vista_ppv_png = save_ppv_curve_plot_and_tsv(vista_ppv, vista_ppv_prefix)

print("VISTA AUROC:", vista_auroc)
print("VISTA AUPRC:", vista_auprc)
print("VISTA max PPV (TP>=min_tp):", vista_ppv["best_ppv"])
print("VISTA best_k:", vista_ppv["best_k"])
print("VISTA best_threshold:", vista_ppv["best_threshold"])
print("Saved VISTA PPV curve:", vista_ppv_tsv, vista_ppv_png)

wandb.log({
    "vista/AUROC": vista_auroc,
    "vista/AUPRC": vista_auprc,
    f"vista/max_ppv_minTP{params['min_tp_for_max_ppv']}": vista_ppv["best_ppv"],
    f"vista/best_k_minTP{params['min_tp_for_max_ppv']}": (vista_ppv["best_k"] if vista_ppv["best_k"] is not None else -1),
    f"vista/best_threshold_minTP{params['min_tp_for_max_ppv']}": (vista_ppv["best_threshold"] if vista_ppv["best_threshold"] is not None else np.nan),
    "vista/ppv_curve_plot": wandb.Image(vista_ppv_png),
})

vista_eval_art = wandb.Artifact(name=f"{os.path.basename(model_ID_output)}_vista_eval", type="eval")
vista_eval_art.add_file(vista_ppv_tsv)
vista_eval_art.add_file(vista_ppv_png)
wandb.log_artifact(vista_eval_art)

print("\nDone.\n")
wandb.finish()