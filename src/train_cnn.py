from scripts.utils_cnn import load_datasets_from_zip
from scripts.xception_model import build_model, FineTuneCallback
import argparse
import tensorflow as tf
import json
import os
import numpy as np
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main(config):
    print("📦 Loading datasets...")
    train_ds, val_ds, label_to_index, index_to_label, train_labels = load_datasets_from_zip(
        zip_path=config['zip_path'],
        img_size=(config['img_size'], config['img_size']),
        batch_size=config['batch_size'],
        split_ratio=config['split_ratio']
    )

    # Compute class weights --> This is part of dealing with the class imbalance
    labels = [label_to_index[label] for label in train_labels]
    classes = np.array(list(label_to_index.values()))
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weight_dict = dict(zip(classes, class_weights))

    input_shape = (config['img_size'], config['img_size'], 3)
    num_classes = len(label_to_index)

    os.makedirs(config['output_dir'], exist_ok=True)

    cb_list = []
    cb_cfg = config.get("callbacks", {})

    # Checkpoint callback
    if cb_cfg.get("checkpoint", {}).get("enabled", True):
        ckpt_name = cb_cfg["checkpoint"].get("filename", "model_checkpoint.keras")
        save_best = cb_cfg["checkpoint"].get("save_best_only", True)
        checkpoint_cb = callbacks.ModelCheckpoint(
            os.path.join(config['output_dir'], ckpt_name),
            save_best_only=save_best
        )
        cb_list.append(checkpoint_cb)

    # EarlyStopping callback
    if cb_cfg.get("early_stopping", {}).get("enabled", True):
        early_cb = callbacks.EarlyStopping(
            patience=cb_cfg["early_stopping"].get("patience", 5),
            restore_best_weights=cb_cfg["early_stopping"].get("restore_best_weights", True)
        )
        cb_list.append(early_cb)

    # TensorBoard callback
    if cb_cfg.get("tensorboard", {}).get("enabled", True):
        tb_log_dir = os.path.join(config['output_dir'], cb_cfg["tensorboard"].get("log_dir", "logs"))
        cb_list.append(callbacks.TensorBoard(log_dir=tb_log_dir))

    checkpoint_path = os.path.join(config['output_dir'], "model_checkpoint.keras")
    logs_dir = os.path.join(config['output_dir'], "logs")

    # Load or initialize model
    if config.get('resume') and os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
        base_model = model.layers[0]  # Assuming base model is the first layer
    else:
        print("🧠 Building new model...")
        model, base_model = build_model(input_shape, num_classes)

    # Optional: fine-tune after N epochs
    if config.get('fine_tune_at') is not None:
        cb_list.append(FineTuneCallback(
            base_model=base_model,
            unfreeze_epoch=config['fine_tune_at'],
            learning_rate=config.get('fine_tune_lr', 1e-5)
        ))

    print("🚀 Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=cb_list,
        class_weight=class_weight_dict
    )

    final_model_path = os.path.join(config['output_dir'], "final_model.keras")
    print(f"💾 Saving final model to {final_model_path}")
    model.save(final_model_path)

    print("📊 Evaluating model on validation set...")
    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=[index_to_label[i] for i in sorted(index_to_label)])

    report_path = os.path.join(config['output_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Classification report saved to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    writer = tf.summary.create_file_writer(os.path.join(config['output_dir'], 'logs', 'eval'))
    with writer.as_default():
        for i, acc in enumerate(per_class_acc):
            tf.summary.scalar(f"eval_accuracy/{index_to_label[i]}", acc, step=0)
            print(f"{index_to_label[i]:<35} - Accuracy: {acc:.2%}")

    config_save_path = os.path.join(config['output_dir'], "used_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Training complete. View logs with: tensorboard --logdir", logs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Xception model on plankton images")
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    # CLI overrides
    parser.add_argument("--zip_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--split_ratio", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")
    parser.add_argument("--fine_tune_at", type=int)
    parser.add_argument("--fine_tune_lr", type=float)

    args = parser.parse_args()

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

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 Using GPU: {gpus[0].name}")
    else:
        print("⚠️ No GPU detected. Training will be slower.")

    main(config)