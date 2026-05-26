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

DOG_OPTIONS = (
    "n02094433",  # Yorkshire terrier
    "n02099601",  # golden retriever
    "n02099712",  # Labrador retriever
    "n02106662",  # German shepherd
    "n02113799",  # standard poodle
)

CAT_OPTIONS = (
    "n02123045",  # tabby
    "n02123394",  # Persian cat
    "n02124075",  # Egyptian cat
    "n02125311",  # cougar
    "n02129165",  # lion
)

CAT_TRUTH_POOL = (
    "n02123045",  # tabby
    "n02123394",  # Persian cat
    "n02124075",  # Egyptian cat
)

CAT_TRUTH_ORDER = (
    "n02123045",
    "n02123394",
    "n02124075",
    "n02123045",
    "n02123394",
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
    for wnid in DOG_OPTIONS + CAT_OPTIONS:
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


def load_imagenet_class_index() -> dict[str, int]:
    cache_path = BASE_DIR / ".torch-cache" / "imagenet_class_index.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, cache_path)
    with cache_path.open("r", encoding="utf-8") as handle:
        class_index = json.load(handle)
    return {value[0]: int(index) for index, value in class_index.items()}


def score_all_images(
    classes: dict[str, dict[str, Any]],
    wnid_to_index: dict[str, int],
) -> dict[Path, dict[str, Any]]:
    """For every training image of every candidate class, compute the softmax
    score Inception assigns to each option WNID and the model's top-1 class.
    Returns image_path -> {"option_probs": [...], "top1": int}."""
    import torch
    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = inception_v3(weights=weights, aux_logits=True)
    model.eval()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    option_wnids = list(DOG_OPTIONS + CAT_OPTIONS)
    option_indices = torch.tensor([wnid_to_index[w] for w in option_wnids])

    scores: dict[Path, dict[str, Any]] = {}
    all_images: list[Path] = []
    for wnid in option_wnids:
        all_images.extend(classes[wnid]["images"])

    batch_size = 32
    for start in range(0, len(all_images), batch_size):
        chunk = all_images[start:start + batch_size]
        tensors = []
        for path in chunk:
            with Image.open(path).convert("RGB") as image:
                tensors.append(preprocess(image))
        batch = torch.stack(tensors)
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            option_probs = probs.index_select(1, option_indices)
            top1 = probs.argmax(dim=1)
        for path, row, top in zip(chunk, option_probs, top1):
            scores[path] = {
                "option_probs": [float(v) for v in row],
                "top1": int(top),
            }
        print(f"  scored {min(start + batch_size, len(all_images))}/{len(all_images)} images", flush=True)
    return scores


def margin_for(
    image: Path,
    truth_wnid: str,
    option_wnids: tuple[str, ...],
    scores: dict[Path, dict[str, Any]],
    order: list[str],
) -> float:
    """Own-class prob minus best-other-option prob, restricted to the option set."""
    row = scores[image]["option_probs"]
    own = row[order.index(truth_wnid)]
    best_other = max(
        row[order.index(w)] for w in option_wnids if w != truth_wnid
    )
    return own - best_other


def recognized_images(
    images: tuple[Path, ...],
    own_wnid: str,
    scores: dict[Path, dict[str, Any]],
    wnid_to_index: dict[str, int],
) -> list[Path]:
    """Keep only images whose top-1 ImageNet prediction is the own class.
    These are genuinely class-recognizable images by the model."""
    own_index = wnid_to_index[own_wnid]
    return [img for img in images if scores[img]["top1"] == own_index]


def pick_focal_image(
    truth: dict[str, Any],
    option_wnids: tuple[str, ...],
    scores: dict[Path, dict[str, Any]],
    order: list[str],
    wnid_to_index: dict[str, int],
    rng: random.Random,
    used: set[Path],
) -> Path:
    """Pick the most-confusable training image of the truth class. Restrict to
    images Inception recognizes (top-1 == truth) so the image is unambiguously
    of the truth class, then take the smallest within-option margin."""
    available = tuple(img for img in truth["images"] if img not in used)
    recognized = recognized_images(available, truth["wnid"], scores, wnid_to_index)
    if not recognized:
        recognized = list(available)
    ranked = sorted(
        recognized,
        key=lambda p: margin_for(p, truth["wnid"], option_wnids, scores, order),
    )
    pool_size = max(1, len(ranked) // 10)
    pool = ranked[:pool_size]
    return rng.choice(pool)


def pick_exemplars(
    option_class: dict[str, Any],
    option_wnids: tuple[str, ...],
    scores: dict[Path, dict[str, Any]],
    order: list[str],
    wnid_to_index: dict[str, int],
    rng: random.Random,
    used: set[Path],
) -> list[Path]:
    """Pick exemplars for a candidate class. Restrict to images Inception
    recognizes as the class (top-1 == own) so they unambiguously belong,
    then prefer the lowest-margin ones (least textbook-prototypical)."""
    available = tuple(img for img in option_class["images"] if img not in used)
    recognized = recognized_images(available, option_class["wnid"], scores, wnid_to_index)
    if len(recognized) < EXAMPLES_PER_CLASS:
        recognized = list(available)
    ranked = sorted(
        recognized,
        key=lambda p: margin_for(p, option_class["wnid"], option_wnids, scores, order),
    )
    pool_size = max(EXAMPLES_PER_CLASS * 3, len(ranked) // 5)
    pool = ranked[:pool_size]
    return rng.sample(pool, EXAMPLES_PER_CLASS)


def make_tasks(
    classes: dict[str, dict[str, Any]],
    scores: dict[Path, list[float]],
) -> list[dict[str, Any]]:
    rng = random.Random(TASK_SAMPLE_SEED)
    order = list(DOG_OPTIONS + CAT_OPTIONS)

    dog_truth_order = list(DOG_OPTIONS)
    rng.shuffle(dog_truth_order)

    cat_truth_order = list(CAT_TRUTH_ORDER)

    species_plan: list[tuple[str, str, tuple[str, ...]]] = []
    dog_iter = iter(dog_truth_order)
    cat_iter = iter(cat_truth_order)
    for index in range(TASK_COUNT):
        species = "dog" if index % 2 == 0 else "cat"
        if species == "dog":
            truth_wnid = next(dog_iter)
            options = DOG_OPTIONS
        else:
            truth_wnid = next(cat_iter)
            options = CAT_OPTIONS
        species_plan.append((species, truth_wnid, options))

    used: set[Path] = set()
    tasks: list[dict[str, Any]] = []
    wnid_to_index = load_imagenet_class_index()
    for index, (species, truth_wnid, option_wnids) in enumerate(species_plan):
        truth = classes[truth_wnid]
        focal = pick_focal_image(truth, option_wnids, scores, order, wnid_to_index, rng, used)
        used.add(focal)

        shuffled = list(option_wnids)
        rng.shuffle(shuffled)

        task_options = []
        for wnid in shuffled:
            option_class = classes[wnid]
            exemplars = pick_exemplars(option_class, option_wnids, scores, order, wnid_to_index, rng, used)
            used.update(exemplars)
            task_options.append(
                {
                    "wnid": wnid,
                    "label": option_class["label"],
                    "examples": [static_image_path(image) for image in exemplars],
                }
            )

        tasks.append(
            {
                "index": index,
                "species": species,
                "image": static_image_path(focal),
                "truth": truth_wnid,
                "options": task_options,
            }
        )
    return tasks


def add_ai_predictions(tasks: list[dict[str, Any]], scores: dict[Path, dict[str, Any]], root: Path, wnid_to_index: dict[str, int]) -> None:
    from torchvision.models import Inception_V3_Weights

    categories = list(Inception_V3_Weights.DEFAULT.meta["categories"])
    order = list(DOG_OPTIONS + CAT_OPTIONS)
    for task in tasks:
        focal_path = source_from_static_path(root, task["image"])
        entry = scores[focal_path]
        focal_row = entry["option_probs"]
        option_scores: dict[str, float] = {}
        for option in task["options"]:
            option_scores[option["wnid"]] = focal_row[order.index(option["wnid"])]
        choice = max(option_scores, key=option_scores.get)
        top_index = entry["top1"]

        task["ai"] = {
            "choice": choice,
            "confidence": option_scores[choice],
            "scores": option_scores,
            "topImagenetLabel": categories[top_index],
        }


def source_from_static_path(root: Path, static_path: str) -> Path:
    _, wnid, filename = Path(static_path).parts
    return root / "train" / wnid / "images" / filename


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
    wnid_to_index = load_imagenet_class_index()
    print(f"Scoring {sum(len(classes[w]['images']) for w in DOG_OPTIONS + CAT_OPTIONS)} candidate images...", flush=True)
    scores = score_all_images(classes, wnid_to_index)
    tasks = make_tasks(classes, scores)
    add_ai_predictions(tasks, scores, root, wnid_to_index)
    image_count = copy_referenced_images(tasks, root, docs_dir)
    write_tasks(tasks, docs_dir)
    print(f"Wrote {len(tasks)} tasks to {docs_dir / 'tasks.json'}")
    print(f"Copied {image_count} images into {docs_dir / 'images'}")


if __name__ == "__main__":
    main()
