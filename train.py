"""Train the CNN + additive-attention LSTM captioner on Flickr8k captions."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from caption_model import AttentionLSTMCaptioner


PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def read_captions(captions_file: Path):
    pairs = []
    with captions_file.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        if captions_file.suffix.lower() == ".token" or "\t" in first_line:
            for row in f:
                parts = row.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    pairs.append((parts[0].split("#", 1)[0], parts[1]))
        else:
            for row in csv.reader(f):
                if len(row) < 2 or row[0].strip().lower() in {"image", "image_name", "filename"}:
                    continue
                pairs.append((row[0].strip(), ",".join(row[1:]).strip()))
    if not pairs:
        raise ValueError(f"No image-caption pairs found in {captions_file}")
    return pairs


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def image_path(images_dir: Path, name: str):
    path = images_dir / name
    if path.is_file():
        return path
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / (name + extension)
        if candidate.is_file():
            return candidate
    return None


def make_vocabulary(pairs, min_frequency: int):
    counts = Counter(word for _, caption in pairs for word in tokenize(caption))
    words = [PAD, UNK, START, END]
    words.extend(word for word, count in counts.items() if count >= min_frequency)
    return {word: index for index, word in enumerate(words)}


def cache_image_features(images_dir: Path, names, cache_dir: Path, device, batch_size: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights = ResNet50_Weights.DEFAULT
    backbone = resnet50(weights=weights).to(device).eval()
    encoder = nn.Sequential(*list(backbone.children())[:-2]).to(device).eval()
    del backbone
    transform = weights.transforms()
    batch_names, batch_pixels = [], []
    processed = sum(
        1 for name in names if (cache_dir / f"{Path(name).stem}.pt").exists()
    )

    with torch.inference_mode():
        def save_batch():
            nonlocal processed, batch_names, batch_pixels
            if not batch_names:
                return
            pixel_batch = torch.stack(batch_pixels).to(device)
            feature_maps = encoder(pixel_batch)
            region_features = feature_maps.flatten(2).transpose(1, 2)
            for name, regions in zip(batch_names, region_features):
                cached = cache_dir / f"{Path(name).stem}.pt"
                torch.save(regions.to(dtype=torch.float16, device="cpu"), cached)
            processed += len(batch_names)
            if processed % 250 < len(batch_names):
                print(f"Cached CNN features for {processed}/{len(names)} images", flush=True)
            batch_names, batch_pixels = [], []

        for index, name in enumerate(names, start=1):
            cached = cache_dir / f"{Path(name).stem}.pt"
            if cached.exists():
                continue
            source = image_path(images_dir, name)
            if source is None:
                continue
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                batch_pixels.append(transform(image))
            batch_names.append(name)
            if len(batch_names) >= batch_size:
                save_batch()
        save_batch()
    del encoder


class CaptionDataset(Dataset):
    def __init__(self, pairs, images_dir, cache_dir, vocabulary):
        self.rows = []
        self.cache_dir = cache_dir
        self.vocabulary = vocabulary
        for name, caption in pairs:
            if image_path(images_dir, name) is None:
                continue
            tokens = [START, *tokenize(caption), END]
            ids = [vocabulary.get(word, vocabulary[UNK]) for word in tokens]
            self.rows.append((Path(name).stem, ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        stem, ids = self.rows[index]
        feature = torch.load(self.cache_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
        return feature.float(), torch.tensor(ids, dtype=torch.long)


def collate_batch(batch):
    features, captions = zip(*batch)
    max_length = max(len(caption) for caption in captions)
    padded = torch.zeros((len(captions), max_length), dtype=torch.long)
    for row, caption in enumerate(captions):
        padded[row, : len(caption)] = caption
    return torch.stack(features), padded


def run_epoch(model, loader, optimizer, criterion, device, training):
    model.train(training)
    loss_sum = 0.0
    item_count = 0
    start_time = time.time()
    for batch_index, (features, captions) in enumerate(loader, start=1):
        features, captions = features.to(device), captions.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, captions)
            targets = captions[:, 1 : 1 + logits.size(1)]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        loss_sum += loss.item() * features.size(0)
        item_count += features.size(0)
        if batch_index % 100 == 0:
            phase = "train" if training else "validation"
            print(
                f"  {phase} batch {batch_index}/{len(loader)} "
                f"({time.time() - start_time:.0f}s elapsed)",
                flush=True,
            )
    return loss_sum / max(1, item_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder containing the Flickr8k image files")
    parser.add_argument("--captions", type=Path, required=True, help="Flickr8k.token or CSV captions file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/flickr8k_attention.pt"))
    parser.add_argument("--resume", type=Path, help="Continue training from an existing captioner checkpoint")
    parser.add_argument("--cache", type=Path, default=Path(".feature_cache"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    # Keep the pretrained ResNet weights in the writable project workspace.
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".model_cache" / "hub"))
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    pairs = read_captions(args.captions)
    names = sorted({name for name, _ in pairs})
    split = max(1, int(len(names) * 0.9))
    train_names, valid_names = set(names[:split]), set(names[split:])
    train_pairs = [(name, text) for name, text in pairs if name in train_names]
    valid_pairs = [(name, text) for name, text in pairs if name in valid_names]
    vocabulary = make_vocabulary(train_pairs, args.min_frequency)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_image_features(args.images, names, args.cache, device, args.feature_batch_size)
    train_data = CaptionDataset(train_pairs, args.images, args.cache, vocabulary)
    valid_data = CaptionDataset(valid_pairs, args.images, args.cache, vocabulary)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if resume_checkpoint["vocabulary"] != vocabulary:
            raise ValueError("The resume checkpoint vocabulary does not match the supplied captions.")
        args.embedding_dim = resume_checkpoint["embedding_dim"]
        args.hidden_dim = resume_checkpoint["hidden_dim"]
        args.attention_dim = resume_checkpoint["attention_dim"]

    model = AttentionLSTMCaptioner(
        len(vocabulary),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary[PAD])
    start_epoch = 0
    best_loss = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        # Re-evaluate the saved weights so older checkpoints without validation
        # metadata can still be resumed without discarding a better model.
        best_loss = run_epoch(model, valid_loader, optimizer, criterion, device, training=False)
        print(f"Resuming at epoch {start_epoch}; checkpoint validation loss {best_loss:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; training captions: {len(train_data)}; validation captions: {len(valid_data)}", flush=True)
    print(
        f"Decoder dimensions: embedding={args.embedding_dim}, hidden={args.hidden_dim}, "
        f"attention={args.attention_dim}; batch size={args.batch_size}",
        flush=True,
    )
    print("Architecture: ResNet50 spatial CNN encoder -> additive attention -> LSTM decoder")
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        valid_loss = run_epoch(model, valid_loader, optimizer, criterion, device, training=False)
        print(
            f"Epoch {epoch}/{start_epoch + args.epochs} - train loss {train_loss:.4f} "
            f"- validation loss {valid_loss:.4f}",
            flush=True,
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabulary": vocabulary,
                    "architecture": "resnet50-additive-attention-lstm",
                    "feature_dim": 2048,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "attention_dim": args.attention_dim,
                    "epoch": epoch,
                    "best_validation_loss": best_loss,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
