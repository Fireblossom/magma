import torch
from tqdm import tqdm
from .utils import reduce_losses, to_cuda_half
from torchvision.utils import make_grid
from transformers.tokenization_utils_base import BatchEncoding


def train_step(config, train_loader, model_engine):
    losses = []

    for _ in range(config.gradient_accumulation_steps):
        images, captions = next(train_loader)
        images, captions = to_cuda_half(images, captions)
        if config.run_blind:
            images = torch.zeros_like(images)
        outputs = model_engine(images, captions)
        loss = outputs.loss
        losses.append(loss)
        model_engine.backward(loss)
        model_engine.step()

    return reduce_losses(torch.mean(torch.stack(losses))).item()

"""
def train_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []
    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)
        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())
        model_engine.backward(loss)
        model_engine.step()

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced
"""

def eval_step(config, eval_loader, model_engine):
    losses = []

    for i in tqdm(range(config.eval_steps), "evaluating..."):
        images, captions = next(eval_loader)
        images, captions = to_cuda_half(images, captions)
        if config.run_blind:
            images = torch.zeros_like(images)
        outputs = model_engine(images, captions)
        loss = outputs.loss
        losses.append(loss)

    return reduce_losses(torch.mean(torch.stack(losses))).item()

"""
def eval_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []
    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)
        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced
"""

def inference_step(config, eval_loader, model_engine):
    images, _ = next(eval_loader)
    images = to_cuda_half(images)
    if config.run_blind:
        images = torch.zeros_like(images)
    captions = model_engine(
        images, captions=None, inference=True, ref=True,
    )  # [caption1, caption2, ... b]
    if isinstance(images, BatchEncoding):
        width = min(2, images['pixel_values'].shape[0])
        image_grid = make_grid(images['pixel_values'][:width])
    else:
        width = min(2, images.shape[0])
        image_grid = make_grid(images[:width])
    caption = ""
    for i in range(width):
        caption += f"Caption {i}: \n{captions[i]}\n"
    return image_grid, caption


def final_inference_step(config, eval_loader, model_engine):
    images, _ = next(eval_loader)
    images = to_cuda_half(images)
    if config.run_blind:
        images = torch.zeros_like(images)
    captions = model_engine(
        images, captions=None, inference=True, ref=True,
    )  # [caption1, caption2, ... b]
    if isinstance(images, BatchEncoding):
        image_grid = make_grid(images['pixel_values'])
    else:
        image_grid = make_grid(images)

    return image_grid, captions, _