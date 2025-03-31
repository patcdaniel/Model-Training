import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, mixed_precision
from scripts.utils_cnn import load_datasets_from_zip
from scripts.xception_model import build_model, FineTuneCallback, PeriodicSaveCallback


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Xception model on plankton images")
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    # CLI overrides
    parser.add_argument("--zip_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--split_ratio", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--fine_tune_at", type=int)
    parser.add_argument("--fine_tune_lr", type=float)

    return parser.parse_args()

def load_and_merge_config(args):
    config = {}
    if args.config:
        print(f"📖 Loading config from {args.config}")
        config = load_config(args.config)

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    config.update(cli_overrides)

    required = ['zip_path', 'batch_size', 'epochs', 'img_size', 'split_ratio', 'output_dir']
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config values: {missing}")

    return config

def evaluate_model(model, val_ds, index_to_label, output_dir):
    print("📊 Evaluating model on validation set...")
    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=[index_to_label[i] for i in sorted(index_to_label)])

    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Classification report saved to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    writer = tf.summary.create_file_writer(os.path.join(output_dir, 'logs', 'eval'))
    with writer.as_default():
        for i, acc in enumerate(per_class_acc):
            tf.summary.scalar(f"eval_accuracy/{index_to_label[i]}", acc, step=0)
            print(f"{index_to_label[i]:<35} - Accuracy: {acc:.2%}")

def train_model(model, train_ds, val_ds, config, cb_list, class_weight_dict, initial_epoch, num_train_samples):
    print("🚀 Starting training...")
    steps_per_epoch = (num_train_samples // config['batch_size']) // 2
    validation_steps = steps_per_epoch // 4

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        initial_epoch=initial_epoch,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

def main(config):
    # Config Loading
    if config.get("mixed_precision"):
        mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled:", mixed_precision.global_policy())

    # Dataset Preparation
    print("📦 Loading datasets...")
    train_ds, val_ds, label_to_index, index_to_label, train_labels, num_train_samples = load_datasets_from_zip(
        zip_path=config['zip_path'],
        img_size=(config['img_size'], config['img_size']),
        batch_size=config['batch_size'],
        split_ratio=config['split_ratio']
    )

    labels = [label_to_index[label] for label in train_labels]
    classes = np.array(list(label_to_index.values()))
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weight_dict = dict(zip(classes, class_weights))

    input_shape = (config['img_size'], config['img_size'], 3)
    num_classes = len(label_to_index)

    os.makedirs(config['output_dir'], exist_ok=True)

    # Model Setup
    cb_list = []
    cb_cfg = config.get("callbacks", {})

    if cb_cfg.get("checkpoint", {}).get("enabled", True):
        ckpt_name = cb_cfg["checkpoint"].get("filename", "model_checkpoint.keras")
        save_best = cb_cfg["checkpoint"].get("save_best_only", True)
        checkpoint_cb = callbacks.ModelCheckpoint(
            os.path.join(config['output_dir'], ckpt_name),
            save_best_only=save_best
        )
        cb_list.append(checkpoint_cb)

    if cb_cfg.get("early_stopping", {}).get("enabled", True):
        early_cb = callbacks.EarlyStopping(
            patience=cb_cfg["early_stopping"].get("patience", 5),
            restore_best_weights=cb_cfg["early_stopping"].get("restore_best_weights", True)
        )
        cb_list.append(early_cb)

    if cb_cfg.get("tensorboard", {}).get("enabled", True):
        tb_log_dir = os.path.join(config['output_dir'], cb_cfg["tensorboard"].get("log_dir", "logs"))
        cb_list.append(callbacks.TensorBoard(log_dir=tb_log_dir))
        
    if config.get("save_every_n_epochs"):
        cb_list.append(PeriodicSaveCallback(
            output_dir=config["output_dir"],
            every_n_epochs=config["save_every_n_epochs"],
            prefix="checkpoint"
        ))

    checkpoint_path = os.path.join(config['output_dir'], "model_checkpoint.keras")

    if config.get('resume') and os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
        base_model = model.layers[0]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('fine_tune_lr', 1e-5)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        initial_epoch = config.get('resume_epoch', 0)

        recompile_flag = os.path.join(config["output_dir"], ".recompile_needed")
        if os.path.exists(recompile_flag):
            print("♻️ Detected recompile flag, recompiling model...")
            base_model.trainable = True
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('fine_tune_lr', 1e-5)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            os.remove(recompile_flag)

    else:
        print("🧠 Building new model...")
        model, base_model = build_model(input_shape, num_classes)
        initial_epoch = 0

    model.output_dir = config["output_dir"]

    if config.get('fine_tune_at') is not None:
        cb_list.append(FineTuneCallback(
            base_model=base_model,
            unfreeze_epoch=config['fine_tune_at'],
            learning_rate=config.get('fine_tune_lr', 1e-5)
        ))

    train_model(model, train_ds, val_ds, config, cb_list, class_weight_dict, initial_epoch, num_train_samples)

    evaluate_model(model, val_ds, index_to_label, config['output_dir'])
    
    label_map_path = os.path.join(config['output_dir'], "label_to_index.json")
    with open(label_map_path, 'w') as f:
        json.dump(label_to_index, f, indent=2)
    print(f"🗂️ Label mapping saved to {label_map_path}")

    config["resume_epoch"] = config["epochs"]

    config_save_path = os.path.join(config['output_dir'], "used_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete. View logs with: tensorboard --logdir", os.path.join(config['output_dir'], 'logs'))

if __name__ == "__main__":
    args = parse_args()
    config = load_and_merge_config(args)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 Using GPU: {gpus[0].name}")
    else:
        print("⚠️ No GPU detected. Training will be slower.")

    main(config)