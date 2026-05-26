from __future__ import annotations

import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_DEPS = BASE_DIR / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

os.environ.setdefault("TORCH_HOME", str(BASE_DIR / ".torch-cache"))

TASK_COUNT = 10
OPTION_COUNT = 5
EXAMPLES_PER_CLASS = 5
TASK_SAMPLE_SEED = 20260526
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

DOG_WNIDS = (
    "n02085620",
    "n02094433",
    "n02099601",
    "n02099712",
    "n02106662",
    "n02113799",
)

CAT_WNIDS = (
    "n02123045",
    "n02123394",
    "n02124075",
    "n02125311",
    "n02129165",
)


def dataset_root() -> Path:
    candidates = [
        Path(os.environ["TINY_IMAGENET_DIR"]) if os.environ.get("TINY_IMAGENET_DIR") else None,
        BASE_DIR / "tiny-imagenet-200",
        BASE_DIR / "tiny-image-net",
        Path("/tiny-image-net"),
        Path("/tiny-imagenet-200"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if (candidate / "train").is_dir():
            return candidate
        if candidate.name == "train" and candidate.is_dir():
            return candidate.parent
    raise RuntimeError("Tiny ImageNet dataset was not found.")


def display_label(label: str) -> str:
    primary = label.split(",")[0].strip()
    return primary[:1].upper() + primary[1:]


def load_words(root: Path) -> dict[str, str]:
    words: dict[str, str] = {}
    with (root / "words.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            wnid, names = line.rstrip("\n").split("\t", 1)
            words[wnid] = names
    return words


def load_classes(root: Path) -> dict[str, dict[str, Any]]:
    words = load_words(root)
    classes: dict[str, dict[str, Any]] = {}
    for wnid in DOG_WNIDS + CAT_WNIDS:
        image_dir = root / "train" / wnid / "images"
        images = tuple(sorted(image_dir.glob("*.JPEG")))
        if len(images) < EXAMPLES_PER_CLASS + 1:
            raise RuntimeError(f"Not enough images for {wnid}.")
        classes[wnid] = {
            "wnid": wnid,
            "label": display_label(words.get(wnid, wnid)),
            "images": images,
        }
    return classes


def static_image_path(image_path: Path) -> str:
    wnid = image_path.parent.parent.name
    return f"images/{wnid}/{image_path.name}"


def option_set_for_task(species_classes: list[dict[str, Any]], truth: dict[str, Any], species_task_index: int) -> list[dict[str, Any]]:
    if len(species_classes) == OPTION_COUNT:
        return species_classes[:]
    candidates = [class_info for class_info in species_classes if class_info["wnid"] != truth["wnid"]]
    omitted = candidates[species_task_index % len(candidates)]
    return [class_info for class_info in species_classes if class_info["wnid"] != omitted["wnid"]]


def make_tasks(classes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(TASK_SAMPLE_SEED)
    grouped = {
        "dog": [classes[wnid] for wnid in DOG_WNIDS],
        "cat": [classes[wnid] for wnid in CAT_WNIDS],
    }
    focal_classes = {
        "dog": rng.sample(grouped["dog"], 5),
        "cat": rng.sample(grouped["cat"], 5),
    }
    species_counts = {"dog": 0, "cat": 0}
    tasks: list[dict[str, Any]] = []

    for index, species in enumerate(["dog", "cat"] * 5):
        species_index = species_counts[species]
        species_counts[species] += 1
        truth = focal_classes[species][species_index]
        focal_image = rng.choice(truth["images"])
        options = option_set_for_task(grouped[species], truth, species_index)
        rng.shuffle(options)
        task_options = []
        for option in options:
            pool = [image for image in option["images"] if image != focal_image]
            examples = rng.sample(pool, EXAMPLES_PER_CLASS)
            task_options.append(
                {
                    "wnid": option["wnid"],
                    "label": option["label"],
                    "examples": [static_image_path(image) for image in examples],
                }
            )
        tasks.append(
            {
                "index": index,
                "species": species,
                "image": static_image_path(focal_image),
                "truth": truth["wnid"],
                "options": task_options,
            }
        )
    return tasks


def load_imagenet_class_index() -> dict[str, int]:
    cache_path = BASE_DIR / ".torch-cache" / "imagenet_class_index.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, cache_path)
    with cache_path.open("r", encoding="utf-8") as handle:
        class_index = json.load(handle)
    return {value[0]: int(index) for index, value in class_index.items()}


def source_from_static_path(root: Path, static_path: str) -> Path:
    _, wnid, filename = Path(static_path).parts
    return root / "train" / wnid / "images" / filename


def add_ai_predictions(tasks: list[dict[str, Any]], root: Path) -> None:
    import torch
    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    categories = list(weights.meta["categories"])
    preprocess = weights.transforms()
    wnid_to_index = load_imagenet_class_index()
    model = inception_v3(weights=weights, aux_logits=True)
    model.eval()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    for task in tasks:
        with Image.open(source_from_static_path(root, task["image"])).convert("RGB") as image:
            batch = preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        scores: dict[str, float] = {}
        for option in task["options"]:
            class_index = wnid_to_index[option["wnid"]]
            scores[option["wnid"]] = float(probs[class_index])
        choice = max(scores, key=scores.get)
        top_index = int(probs.argmax())
        task["ai"] = {
            "choice": choice,
            "confidence": scores[choice],
            "scores": scores,
            "topImagenetLabel": categories[top_index],
            "topImagenetConfidence": float(probs[top_index]),
        }


def copy_referenced_images(tasks: list[dict[str, Any]], root: Path, docs_dir: Path) -> int:
    image_dir = docs_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    referenced: set[str] = set()
    for task in tasks:
        referenced.add(task["image"])
        for option in task["options"]:
            referenced.update(option["examples"])

    for relative in sorted(referenced):
        source = source_from_static_path(root, relative)
        target = docs_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(referenced)


def write_tasks(tasks: list[dict[str, Any]], docs_dir: Path) -> None:
    payload = {
        "taskCount": len(tasks),
        "optionCount": OPTION_COUNT,
        "examplesPerClass": EXAMPLES_PER_CLASS,
        "tasks": tasks,
    }
    with (docs_dir / "tasks.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    root = dataset_root()
    docs_dir = BASE_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    classes = load_classes(root)
    tasks = make_tasks(classes)
    add_ai_predictions(tasks, root)
    image_count = copy_referenced_images(tasks, root, docs_dir)
    write_tasks(tasks, docs_dir)
    print(f"Wrote {len(tasks)} tasks to {docs_dir / 'tasks.json'}")
    print(f"Copied {image_count} images into {docs_dir / 'images'}")


if __name__ == "__main__":
    main()
