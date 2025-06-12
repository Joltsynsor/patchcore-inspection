import sys
sys.path.append('src')

import torch
import patchcore.datasets.masonry as masonry
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

# Set device
device = torch.device("cuda:0")

# Load dataset (training only)
train_dataset = masonry.MasonryDataset(
    source="masonry_dataset",
    classname="wall",
    resize=256,
    imagesize=224,
    split=masonry.DatasetSplit.TRAIN
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

print(f"Training dataset loaded: {len(train_dataset)} images")

# Create PatchCore model
backbone = patchcore.backbones.load("wideresnet50")
backbone.name = "wideresnet50"

sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)
nn_method = patchcore.common.FaissNN(False, 4)  # CPU FAISS

patchcore_model = patchcore.patchcore.PatchCore(device)
patchcore_model.load(
    backbone=backbone,
    layers_to_extract_from=["layer2", "layer3"],
    device=device,
    input_shape=(3, 224, 224),
    pretrain_embed_dimension=1024,
    target_embed_dimension=1024,
    patchsize=3,
    featuresampler=sampler,
    anomaly_scorer_num_nn=1,
    nn_method=nn_method,
)

print("PatchCore model created")

# Train the model
print("Starting training...")
patchcore_model.fit(train_dataloader)
print("Training completed!")

# Save the model
import os
save_path = "results/masonry_model"
os.makedirs(save_path, exist_ok=True)
patchcore_model.save_to_path(save_path)
print(f"Model saved to: {save_path}")

print("Training and saving completed successfully!")
