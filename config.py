from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"
LOG_DIR = ROOT.parent.parent / ".tb_logs" / "sentiment_zh"

SEED = 42
MAX_LEN = 256
VAL_RATIO = 0.1
TRAIN_PER_CLASS = 5000
TEST_PER_CLASS = 500

LSTM_EXPS = {
    "exp1_baseline": {"lr": 1e-3, "batch_size": 64, "dropout": 0.3, "weight_decay": 0.0},
    "exp2_low_lr": {"lr": 1e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 0.0},
    "exp3_large_bs": {"lr": 1e-3, "batch_size": 128, "dropout": 0.3, "weight_decay": 0.0},
    "exp4_regularized": {"lr": 1e-3, "batch_size": 64, "dropout": 0.5, "weight_decay": 1e-4},
}
LSTM_EPOCHS, TRANS_EPOCHS, PROFILE_EPOCHS = 10, 10, 2
