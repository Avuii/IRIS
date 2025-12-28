import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# =========================================================
# RUN / SAVE UTILS
# =========================================================
def init_run(output_root="runs", run_name_prefix="iris"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{run_name_prefix}_{ts}")
    figs_dir = os.path.join(run_dir, "figs")
    tables_dir = os.path.join(run_dir, "tables")
    models_dir = os.path.join(run_dir, "models")

    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    return {"run_dir": run_dir, "figs": figs_dir, "tables": tables_dir, "models": models_dir, "timestamp": ts}

def save_config(run, config: dict, filename="config.json"):
    path = os.path.join(run["run_dir"], filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path

def save_text(run, text: str, filename="notes.txt"):
    path = os.path.join(run["run_dir"], filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def save_csv(run, header, rows, filename="results.csv"):
    path = os.path.join(run["tables"], filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path

def save_model_state(run, model, filename="best_model.pt"):
    path = os.path.join(run["models"], filename)
    torch.save(model.state_dict(), path)
    return path


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean_std(xs):
    xs = np.array(xs, dtype=float)
    return xs.mean(), xs.std()

@torch.no_grad()
def predict_numpy(model, X_np):
    model.eval()
    X = torch.tensor(X_np, dtype=torch.float32)
    logits = model(X)
    return torch.argmax(logits, dim=1).cpu().numpy()

@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    all_y = []
    all_pred = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.shape[0]
        total_n += xb.shape[0]

        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / max(total_n, 1)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred


# =========================================================
# Training with history (for learning curves)
# =========================================================
def train_with_history(model, train_loader, test_loader, epochs=200, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        tr_loss, tr_acc, _, _ = eval_loader(model, train_loader)
        te_loss, te_acc, _, _ = eval_loader(model, test_loader)

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss)
        hist["test_acc"].append(te_acc)

    return hist


# =========================================================
# Plots (save to file)
# =========================================================
def plot_acc_vs_hidden(hidden_sizes, acc_by_h, out_path):
    hs = np.array(hidden_sizes, dtype=float)
    means = np.array([mean_std(acc_by_h[h])[0] for h in hidden_sizes])

    plt.figure(figsize=(7, 4))
    plt.plot(hs, means, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Liczba neuronów w warstwie ukrytej H (log2)")
    plt.ylabel("Średnia accuracy (test)")
    plt.title("IRIS: Accuracy vs liczba neuronów H")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_boxplot_acc(hidden_sizes, acc_by_h, out_path):
    data = [acc_by_h[h] for h in hidden_sizes]

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=[str(h) for h in hidden_sizes], showmeans=True)
    plt.xlabel("H (liczba neuronów w warstwie ukrytej)")
    plt.ylabel("Accuracy (test) po seedach")
    plt.title("IRIS: Rozrzut accuracy vs H (boxplot)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_learning_curves(histories, out_path):
    plt.figure(figsize=(8, 5))
    for name, h in histories.items():
        plt.plot(h["train_loss"], label=f"{name} train")
        plt.plot(h["test_loss"], linestyle="--", label=f"{name} test")

    plt.xlabel("Epoka")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("IRIS: Krzywe uczenia (loss) – train vs test")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predykcja")
    plt.ylabel("Prawda")
    plt.xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_cm_txt(run, cm: np.ndarray, class_names, filename):
    path = os.path.join(run["tables"], filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Confusion matrix\n")
        f.write("Classes: " + ", ".join(class_names) + "\n\n")
        f.write(np.array2string(cm))
        f.write("\n")
    return path


# =========================================================
# Optional WOW: PCA decision regions (train a 2D classifier on PCA space)
# =========================================================
def plot_pca_decision_regions(model_factory, hidden=None, seed=0, out_path="iris_pca_decision_regions.png"):
    set_seed(seed)

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    pca = PCA(n_components=2, random_state=seed)
    Xtr2 = pca.fit_transform(X_train_s).astype(np.float32)
    Xte2 = pca.transform(X_test_s).astype(np.float32)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(Xtr2), torch.tensor(y_train))
    test_ds  = torch.utils.data.TensorDataset(torch.tensor(Xte2), torch.tensor(y_test))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    model = model_factory(in_dim=2, out_dim=3, hidden=hidden)
    _ = train_with_history(model, train_loader, test_loader, epochs=250, lr=0.03)

    x_min, x_max = Xtr2[:, 0].min() - 1.0, Xtr2[:, 0].max() + 1.0
    y_min, y_max = Xtr2[:, 1].min() - 1.0, Xtr2[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    pred = predict_numpy(model, grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, pred, alpha=0.35)
    plt.scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_train, marker="o", label="train")
    plt.scatter(Xte2[:, 0], Xte2[:, 1], c=y_test, marker="^", label="test")

    plt.title(f"IRIS: Granice decyzyjne PCA-2D ({'linear' if hidden is None else f'MLP H={hidden}'})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =========================================================
# Models
# =========================================================
class LinearSoftmax(nn.Module):
    def __init__(self, in_dim=4, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_dim=4, hidden=16, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def model_factory(in_dim, out_dim, hidden=None):
    if hidden is None:
        return LinearSoftmax(in_dim=in_dim, out_dim=out_dim)
    return MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)


# =========================================================
# Data loaders
# =========================================================
def make_loaders(seed, test_size=0.2, batch_size=16):
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t  = torch.tensor(X_test)
    y_test_t  = torch.tensor(y_test)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_ds  = torch.utils.data.TensorDataset(X_test_t, y_test_t)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test, y_test, iris.target_names.tolist()


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def main():
    # --- config (to zapisze się do config.json)
    config = {
        "test_size": 0.2,
        "seeds": [0, 1, 2, 3, 4],
        "hidden_sizes": [1, 2, 4, 8, 16, 32, 64],
        "epochs": 250,
        "lr": 0.03,
        "batch_size": 16,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss"
    }

    run = init_run(output_root="runs", run_name_prefix="iris")
    save_config(run, config)

    seeds = config["seeds"]
    hidden_sizes = config["hidden_sizes"]

    # zbieramy acc po seedach
    acc_linear = []
    acc_by_h = {h: [] for h in hidden_sizes}

    # learning curves tylko z seeda=0 dla 3 modeli
    histories = {}

    # confusion matrices seed=0
    cm_linear_seed0 = None
    cm_best_seed0 = None
    best_h = None

    # do wyboru najlepszego H po mean accuracy
    # (trzymamy też model z seeda 0 dla najlepszego H żeby ewentualnie zapisać wagi)
    best_mean = -1.0
    best_model_seed0 = None

    # per-seed tabela (fajna do aneksu)
    per_seed_rows = []

    for seed in seeds:
        set_seed(seed)
        train_loader, test_loader, X_test_np, y_test_np, class_names = make_loaders(
            seed, test_size=config["test_size"], batch_size=config["batch_size"]
        )

        # ---- baseline linear
        lin = LinearSoftmax(in_dim=4, out_dim=3)
        hist_lin = train_with_history(lin, train_loader, test_loader, epochs=config["epochs"], lr=config["lr"])
        _, acc, _, y_pred = eval_loader(lin, test_loader)
        acc_linear.append(acc)
        per_seed_rows.append(["linear", "", seed, acc])

        if seed == 0:
            histories["linear"] = hist_lin
            cm_linear_seed0 = confusion_matrix(y_test_np, y_pred)

        # ---- MLP for each H
        for h in hidden_sizes:
            mlp = MLP(in_dim=4, hidden=h, out_dim=3)
            hist = train_with_history(mlp, train_loader, test_loader, epochs=config["epochs"], lr=config["lr"])
            _, acc, _, y_pred = eval_loader(mlp, test_loader)

            acc_by_h[h].append(acc)
            per_seed_rows.append(["mlp", h, seed, acc])

            # learning curves tylko dla seed=0 i H=8,32 (jak chciałaś)
            if seed == 0 and h in (8, 32):
                histories[f"mlp_h={h}"] = hist

            # żeby mieć CM dla najlepszego H (po średniej) zapisujemy na końcu,
            # ale tu zapamiętamy model seed=0, bo CM ma być seed=0
            if seed == 0:
                # chwilowo tylko kandydaci; finalny best wybierzemy po mean po seedach
                pass

    # wybór najlepszego H po mean accuracy
    for h in hidden_sizes:
        m = np.mean(acc_by_h[h])
        if m > best_mean or (abs(m - best_mean) < 1e-12 and (best_h is None or h < best_h)):
            best_mean = m
            best_h = h

    # policz CM dla best_h na seed=0 (trenujemy jeszcze raz tylko ten model na seed=0)
    set_seed(0)
    train_loader, test_loader, X_test_np, y_test_np, class_names = make_loaders(
        0, test_size=config["test_size"], batch_size=config["batch_size"]
    )
    best_model_seed0 = MLP(in_dim=4, hidden=best_h, out_dim=3)
    _ = train_with_history(best_model_seed0, train_loader, test_loader, epochs=config["epochs"], lr=config["lr"])
    _, _, _, y_pred_best = eval_loader(best_model_seed0, test_loader)
    cm_best_seed0 = confusion_matrix(y_test_np, y_pred_best)

    # --- zapis tabel
    # summary mean/std
    summary_rows = []
    summary_rows.append(["linear", "", np.mean(acc_linear), np.std(acc_linear)])
    for h in hidden_sizes:
        summary_rows.append(["mlp", h, np.mean(acc_by_h[h]), np.std(acc_by_h[h])])

    save_csv(run, ["model", "H", "acc_mean", "acc_std"], summary_rows, filename="results_summary.csv")
    save_csv(run, ["model", "H", "seed", "acc_test"], per_seed_rows, filename="results_per_seed.csv")

    # --- zapis wykresów (4 rysunki)
    plot_acc_vs_hidden(hidden_sizes, acc_by_h, out_path=os.path.join(run["figs"], "acc_vs_hidden.png"))
    plot_boxplot_acc(hidden_sizes, acc_by_h, out_path=os.path.join(run["figs"], "boxplot_acc_vs_hidden.png"))
    plot_learning_curves(histories, out_path=os.path.join(run["figs"], "learning_curves_loss.png"))

    plot_confusion_matrix(
        cm_linear_seed0, class_names,
        title="Confusion matrix: linear (seed=0)",
        out_path=os.path.join(run["figs"], "cm_linear_seed0.png")
    )
    save_cm_txt(run, cm_linear_seed0, class_names, filename="cm_linear_seed0.txt")

    plot_confusion_matrix(
        cm_best_seed0, class_names,
        title=f"Confusion matrix: best MLP (H={best_h}) (seed=0)",
        out_path=os.path.join(run["figs"], f"cm_best_h{best_h}_seed0.png")
    )
    save_cm_txt(run, cm_best_seed0, class_names, filename=f"cm_best_h{best_h}_seed0.txt")

    # --- zapis informacji o best + model weights
    save_text(
        run,
        text=(
            f"Best hidden size by mean accuracy: H={best_h}\n"
            f"Best mean accuracy (MLP): {best_mean:.6f}\n"
            f"Linear mean accuracy: {np.mean(acc_linear):.6f}\n"
        ),
        filename="best_model_info.txt"
    )
    save_model_state(run, best_model_seed0, filename=f"best_model_mlp_h{best_h}_seed0.pt")

    # --- opcjonalny WOW (PCA decision regions)
    plot_pca_decision_regions(
        model_factory,
        hidden=best_h,
        seed=0,
        out_path=os.path.join(run["figs"], f"pca_decision_regions_best_h{best_h}.png")
    )

    print("Zapisano wszystko do:", run["run_dir"])
    print("Figury:", run["figs"])
    print("Tabele:", run["tables"])
    print("Modele:", run["models"])


if __name__ == "__main__":
    main()
