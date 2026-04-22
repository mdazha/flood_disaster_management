"""Priority-based flood disaster management image classification pipeline."""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import datasets, models, transforms


DATA_DIR = Path("./")
MODEL_NAME = "inception"
# MODEL_NAME = "densenet"
NUM_CLASSES = 3
BATCH_SIZE = 8
NUM_EPOCHS = 15
FEATURE_EXTRACT = True
MODEL_SAVE_PATH = Path("dense_0.pth")

IMAGE_SIZE = 299
IMG_WIDTH = IMAGE_SIZE
IMG_HEIGHT = IMAGE_SIZE
TEST_IMAGE_FOLDER = Path("test")
CONFUSION_MATRIX_PATH = Path("confusematrx.png")
CLASS_NAMES = ("Major_flooding", "Moderate_flooding", "Minor_flooding")

# Project-specific distance heuristics preserved from the original workflow.
DENSENET_DISTANCE_SCORES = [
    4, 7, 10, 6, 2, 8, 4, 7, 7, 4, 4, 3, 1, 4, 9, 3, 6, 2, 9, 8, 6, 6, 4,
    4, 4, 9, 2, 9, 2, 6, 1, 6, 8, 6, 10, 8, 1, 10, 2, 9, 3, 6, 4, 9, 5, 10,
    5, 8, 5,
]

INCEPTION_DISTANCE_SCORES = [
    9, 6, 1, 9, 6, 8, 10, 6, 4, 6, 4, 3, 7, 4, 3, 7, 3, 2, 4, 9, 6, 7, 3, 7,
    1, 6, 6, 8, 3, 5, 1, 3, 3, 10, 4, 9, 1, 7, 4, 7, 1, 10, 4, 6, 10, 3, 1,
    8, 1, 5, 4, 7, 8, 5, 7, 2, 1, 8, 7, 7, 7, 7, 6, 8, 5, 9, 3, 9, 8, 9, 10,
    3, 1, 9, 3, 6, 7, 8, 10, 8, 3, 8, 2, 8, 3, 10, 3, 10, 1, 9, 1, 5, 4, 3,
    3, 1, 8, 7, 1, 9, 2, 7, 7, 10, 2, 10, 7, 5, 10, 8, 5, 10, 4, 8, 1, 7, 1,
    4, 7, 8, 8, 5, 1, 10, 3, 6, 9, 2, 4, 3, 7, 3, 9, 10, 8, 1, 3, 2, 7, 8, 9,
    5, 3, 3, 7, 3, 6, 7, 7, 9, 9, 8, 9, 9, 5, 9, 8, 7, 5, 7, 10,
]


def maybe_mount_google_drive() -> None:
    """Mount Google Drive when the script is executed inside Colab."""
    try:
        from google.colab import drive
    except ImportError:
        return

    drive.mount("/content/drive")


def get_device() -> torch.device:
    """Return the preferred execution device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_pretrained_model(factory, weights_enum_name: str, use_pretrained: bool):
    """Create a torchvision model with backward-compatible pretrained loading."""
    if use_pretrained:
        try:
            weights_enum = getattr(models, weights_enum_name)
            return factory(weights=weights_enum.DEFAULT)
        except (AttributeError, TypeError):
            return factory(pretrained=True)

    try:
        return factory(weights=None)
    except TypeError:
        return factory(pretrained=False)


def set_parameter_requires_grad(model: nn.Module, feature_extracting: bool) -> None:
    """Freeze model parameters when feature extraction is enabled."""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    use_pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """Initialize the requested torchvision model."""
    model_ft: nn.Module
    input_size = 0

    if model_name == "densenet":
        model_ft = build_pretrained_model(
            models.densenet121,
            "DenseNet121_Weights",
            use_pretrained,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "inception":
        model_ft = build_pretrained_model(
            models.inception_v3,
            "Inception_V3_Weights",
            use_pretrained,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_ft, input_size


def build_data_transforms() -> Dict[str, transforms.Compose]:
    """Create the train/validation transforms used by the original workflow."""
    normalization = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    return {
        "train": transforms.Compose(
            [
                transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                transforms.ToTensor(),
                normalization,
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                transforms.ToTensor(),
                normalization,
            ]
        ),
    }


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Build ImageFolder datasets and dataloaders for training and validation."""
    print("Initializing Datasets and Dataloaders...")
    data_transforms = build_data_transforms()
    image_datasets = {
        phase: datasets.ImageFolder(data_dir / phase, data_transforms[phase])
        for phase in ("train", "validation")
    }
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        for phase in ("train", "validation")
    }
    return dataloaders


def get_params_to_update(
    model: nn.Module,
    feature_extract: bool,
) -> List[nn.Parameter]:
    """Collect the parameters that will be optimized."""
    params_to_update = list(model.parameters())
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    return params_to_update


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 25,
    is_inception: bool = False,
) -> Tuple[nn.Module, List[float]]:
    """Train the model and keep the best validation checkpoint."""
    since = time.time()
    val_acc_history: List[float] = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ("train", "validation"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "validation":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def display_sample_images(img_folder: Path, sample_count: int = 10) -> None:
    """Display a small sample of images from the test folder."""
    if not img_folder.exists():
        return

    class_folders = [
        folder_name
        for folder_name in os.listdir(img_folder)
        if (img_folder / folder_name).is_dir()
    ]
    if not class_folders:
        return

    plt.figure(figsize=(20, 20))
    for index in range(sample_count):
        selected_class = random.choice(class_folders)
        image_names = os.listdir(img_folder / selected_class)
        if not image_names:
            continue

        selected_image = random.choice(image_names)
        image_path = img_folder / selected_class / selected_image
        image = mpimg.imread(image_path)

        axis = plt.subplot(1, sample_count, index + 1)
        axis.title.set_text(selected_class)
        plt.imshow(image)

    plt.tight_layout()
    plt.show()


def create_dataset(img_folder: Path) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load and normalize the test dataset into memory."""
    img_data_array: List[np.ndarray] = []
    class_names: List[str] = []
    image_paths: List[str] = []

    for directory_name in os.listdir(img_folder):
        directory_path = img_folder / directory_name
        if not directory_path.is_dir():
            continue

        for file_name in os.listdir(directory_path):
            image_path = directory_path / file_name
            # Preserve the original image-loading behavior while making the intent explicit.
            image = cv2.imread(str(image_path), cv2.IMREAD_ANYCOLOR)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")

            image = cv2.resize(
                image,
                (IMG_HEIGHT, IMG_WIDTH),
                interpolation=cv2.INTER_AREA,
            )
            image = np.array(image, dtype=np.float32)
            image /= 255.0

            image_paths.append(str(image_path))
            img_data_array.append(image)
            class_names.append(directory_name)

    return img_data_array, class_names, image_paths


def im_normalize(image: np.ndarray) -> torch.Tensor:
    """Normalize an image array to match ImageNet-pretrained models."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    tensor = torch.from_numpy(np.array(image)).to(dtype=torch.float32)
    tensor = tensor.permute(-1, 0, 1)
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    return tensor


def predict_dataset(
    model: nn.Module,
    img_data: Sequence[np.ndarray],
    image_paths: Sequence[str],
    device: torch.device,
) -> Tuple[List[int], List[np.ndarray], List[str]]:
    """Run inference on the test dataset."""
    predictions: List[int] = []
    normalized_images: List[np.ndarray] = []
    ordered_paths: List[str] = []

    model.eval()
    with torch.no_grad():
        for index, image in enumerate(img_data):
            normalized_image = im_normalize(image)
            ordered_paths.append(image_paths[index])

            image_tensor = normalized_image.unsqueeze(0).to(device, dtype=torch.float)
            outputs = model(image_tensor).cpu()
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            probability_array = probabilities.detach().numpy()

            print(probability_array)
            predictions.append(int(np.argmax(probability_array)) + 1)
            normalized_images.append(normalized_image.detach().numpy())

    return predictions, normalized_images, ordered_paths


def print_evaluation_metrics(
    title: str,
    targets: Sequence[int],
    predictions: Sequence[int],
) -> None:
    """Print standard classification metrics."""
    print(f"{title} evaluation metrics:")
    print("Accuracy:", accuracy_score(targets, predictions))
    print("F1 Score:", f1_score(targets, predictions, average="macro"))
    print("Precision:", precision_score(targets, predictions, average="macro"))
    print("Recall:", recall_score(targets, predictions, average="macro"))


def print_score_list(scores: Sequence[float]) -> None:
    """Print a one-line numeric score list."""
    for score in scores:
        print(f"{float(score):f}", end=" ")
    print()


def print_path_list(image_paths: Sequence[str]) -> None:
    """Print a path list, one item per line."""
    for image_path in image_paths:
        print(image_path)


def sort_scores_with_paths(
    scores: Sequence[float],
    image_paths: Sequence[str],
) -> Tuple[List[float], List[str]]:
    """Sort image paths in ascending order of score."""
    usable_length = min(len(scores), len(image_paths))
    paired = list(zip(scores[:usable_length], image_paths[:usable_length]))
    paired.sort(key=lambda item: item[0])

    if not paired:
        return [], []

    sorted_scores, sorted_paths = zip(*paired)
    return list(sorted_scores), list(sorted_paths)


def report_ranked_results(
    title: str,
    scores: Sequence[float],
    image_paths: Sequence[str],
) -> Tuple[List[float], List[str]]:
    """Display unsorted and sorted score rankings."""
    usable_length = min(len(scores), len(image_paths))
    scores = list(scores[:usable_length])
    image_paths = list(image_paths[:usable_length])

    print(title)
    print("Given array is")
    print_score_list(scores)
    print_path_list(image_paths)

    sorted_scores, sorted_paths = sort_scores_with_paths(scores, image_paths)
    print("\n\nSorted array is")
    print_score_list(sorted_scores)
    print_path_list(sorted_paths)
    return sorted_scores, sorted_paths


def save_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    output_path: Path,
) -> None:
    """Save and display the confusion matrix."""
    cf_matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(1, NUM_CLASSES + 1)),
    )
    dataframe = pd.DataFrame(cf_matrix, index=CLASS_NAMES, columns=CLASS_NAMES)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig(output_path)
    plt.show()


def main() -> None:
    """Execute training, evaluation, ranking, and visualization."""
    maybe_mount_google_drive()

    model_ft, input_size = initialize_model(
        MODEL_NAME,
        NUM_CLASSES,
        FEATURE_EXTRACT,
        use_pretrained=True,
    )

    print(model_ft)
    print(f"Selected model input size: {input_size}")

    dataloaders_dict = create_dataloaders(DATA_DIR, BATCH_SIZE)
    device = get_device()
    model_ft = model_ft.to(device)

    params_to_update = get_params_to_update(model_ft, FEATURE_EXTRACT)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_ft, _ = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        device=device,
        num_epochs=NUM_EPOCHS,
        is_inception=(MODEL_NAME == "inception"),
    )

    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)

    display_sample_images(TEST_IMAGE_FOLDER)
    img_data, class_names, image_paths = create_dataset(TEST_IMAGE_FOLDER)

    target_dict = {
        class_name: index + 1
        for index, class_name in enumerate(np.unique(class_names))
    }
    print(target_dict)

    target_values = [target_dict[class_names[index]] for index in range(len(class_names))]

    predictions, normalized_images, ordered_paths = predict_dataset(
        model_ft,
        img_data,
        image_paths,
        device,
    )

    if normalized_images:
        plt.imshow(np.einsum("kli->lik", normalized_images[0]))
        plt.show()

    print_evaluation_metrics("Inception V3", target_values, predictions)
    print_evaluation_metrics("Densenet", target_values, predictions)

    new_inception_res = list(predictions)
    report_ranked_results(
        "Prediction-based ranking:",
        [float(score) for score in new_inception_res],
        ordered_paths,
    )

    densenet_weighted_scores = [
        prediction / distance
        for prediction, distance in zip(
            map(float, predictions),
            map(float, DENSENET_DISTANCE_SCORES),
        )
    ]
    report_ranked_results(
        "Densenet distance-adjusted ranking:",
        densenet_weighted_scores,
        ordered_paths,
    )

    inception_weighted_scores = [
        prediction / distance
        for prediction, distance in zip(
            map(float, predictions),
            map(float, INCEPTION_DISTANCE_SCORES),
        )
    ]
    print(predictions)
    report_ranked_results(
        "Inception V3 distance-adjusted ranking:",
        inception_weighted_scores,
        ordered_paths,
    )

    save_confusion_matrix(target_values, predictions, CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
