from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os

from PIL import Image

from style_and_sense.config import MODELS_DIR
from style_and_sense.vocab import (
    COLORS,
    FORMALITY_LEVELS,
    SEASON_TAGS,
    STYLE_TAGS,
    SUBCATEGORIES,
)


FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"


COLOR_RGB = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (125, 125, 125),
    "navy": (20, 35, 75),
    "blue": (45, 95, 180),
    "light blue": (130, 185, 225),
    "brown": (95, 60, 35),
    "tan": (190, 155, 110),
    "cream": (235, 220, 185),
    "green": (55, 130, 75),
    "olive": (95, 110, 55),
    "red": (180, 45, 45),
    "pink": (220, 130, 165),
    "purple": (120, 75, 160),
    "yellow": (220, 190, 55),
    "orange": (220, 120, 45),
    "denim": (70, 105, 150),
}


@dataclass(frozen=True)
class MetadataSuggestion:
    category: str
    subcategory: str
    colors: list[str]
    style_tags: list[str]
    season_tags: list[str]
    formality: str
    caption: str


class MetadataSuggestionError(RuntimeError):
    pass


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def nearest_palette_color(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    distances = {
        name: (red - value[0]) ** 2 + (green - value[1]) ** 2 + (blue - value[2]) ** 2
        for name, value in COLOR_RGB.items()
    }
    return min(distances, key=distances.get)


def extract_dominant_colors(image_bytes: bytes, max_colors: int = 3) -> list[str]:
    image = image_from_bytes(image_bytes)
    image.thumbnail((160, 160))
    quantized = image.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    color_counts = quantized.getcolors(maxcolors=160 * 160) or []

    ranked_colors: list[str] = []
    for _, palette_index in sorted(color_counts, reverse=True):
        offset = palette_index * 3
        rgb = tuple(palette[offset : offset + 3])
        if len(rgb) != 3:
            continue
        color = nearest_palette_color(rgb)
        if color not in ranked_colors:
            ranked_colors.append(color)
        if len(ranked_colors) >= max_colors:
            break

    return ranked_colors or ["black"]


def label_prompt(label: str) -> str:
    return f"a photo of a {label} clothing item"


def top_labels(
    scores_by_label: dict[str, float],
    *,
    limit: int,
    min_score: float = 0.0,
) -> list[str]:
    ranked = sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True)
    return [label for label, score in ranked if score >= min_score][:limit]


class FashionClipTagger:
    def __init__(self, model_name: str = FASHION_CLIP_MODEL) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
        try:
            import torch
            from transformers import AutoModelForZeroShotImageClassification
            from transformers import AutoProcessor
        except ImportError as exc:
            raise MetadataSuggestionError(
                "Install torch and transformers to enable FashionCLIP metadata suggestions."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)
        self.model.eval()

    def score_labels(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        prompts = [label_prompt(label) for label in labels]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)

        return {
            label: float(probs[index].item())
            for index, label in enumerate(labels)
        }

    def suggest(self, image_bytes: bytes, filename: str | None = None) -> MetadataSuggestion:
        image = image_from_bytes(image_bytes)
        subcategory_scores = self.score_labels(image, SUBCATEGORIES)
        style_scores = self.score_labels(image, STYLE_TAGS)
        season_scores = self.score_labels(image, SEASON_TAGS)
        formality_scores = self.score_labels(image, FORMALITY_LEVELS)

        subcategory = top_labels(subcategory_scores, limit=1)[0]
        category = category_for_subcategory(subcategory)
        colors = extract_dominant_colors(image_bytes)
        style_tags = top_labels(style_scores, limit=3)
        season_tags = top_labels(season_scores, limit=2)
        formality = top_labels(formality_scores, limit=1)[0]
        caption = build_caption(
            colors=colors,
            subcategory=subcategory,
            style_tags=style_tags,
            filename=filename,
        )

        return MetadataSuggestion(
            category=category,
            subcategory=subcategory,
            colors=colors,
            style_tags=style_tags,
            season_tags=season_tags,
            formality=formality,
            caption=caption,
        )


def build_caption(
    *,
    colors: list[str],
    subcategory: str,
    style_tags: list[str],
    filename: str | None = None,
) -> str:
    color_text = " and ".join(colors[:2])
    style_text = style_tags[0] if style_tags else ""
    parts = [part for part in [color_text, style_text, subcategory] if part]
    if parts:
        return " ".join(parts)
    return filename or "uploaded garment"


def fallback_metadata_suggestion(
    image_bytes: bytes,
    filename: str | None = None,
) -> MetadataSuggestion:
    colors = extract_dominant_colors(image_bytes)
    subcategory = infer_subcategory_from_filename(filename)
    category = category_for_subcategory(subcategory)
    return MetadataSuggestion(
        category=category,
        subcategory=subcategory,
        colors=colors,
        style_tags=["casual"],
        season_tags=[],
        formality="casual",
        caption=build_caption(
            colors=colors,
            subcategory=subcategory,
            style_tags=["casual"],
            filename=filename,
        ),
    )


def infer_subcategory_from_filename(filename: str | None) -> str:
    if not filename:
        return "t-shirt"
    normalized = filename.lower().replace("-", " ").replace("_", " ")
    for subcategory in SUBCATEGORIES:
        if subcategory in normalized:
            return subcategory
    return "t-shirt"


def category_for_subcategory(subcategory: str) -> str:
    mapping = {
        "t-shirt": "top",
        "tank top": "top",
        "button-down": "top",
        "blouse": "top",
        "sweater": "top",
        "hoodie": "top",
        "jeans": "bottom",
        "trousers": "bottom",
        "shorts": "bottom",
        "skirt": "bottom",
        "dress": "dress",
        "jacket": "outerwear",
        "coat": "outerwear",
        "cardigan": "outerwear",
        "sneakers": "shoes",
        "boots": "shoes",
        "loafers": "shoes",
        "sandals": "shoes",
        "bag": "accessory",
        "hat": "accessory",
        "belt": "accessory",
        "jewelry": "accessory",
    }
    return mapping.get(subcategory, "top")
