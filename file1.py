import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import clip
from tqdm import tqdm
import warnings
import requests
from io import BytesIO
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip

warnings.filterwarnings("ignore")


class EnsembleClassifier:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {device}")
        self.models = {}
        self.processors = {}
        self.tokenizer_openclip = None
        self.setup_models()

    def setup_models(self):
        """Initialize multiple pretrained models for ensemble and DETR for object detection"""
        print("Loading ensemble of models...")

        # 1) OpenAI CLIP ViT-B/32
        try:
            print("Loading OpenAI CLIP ViT-B/32...")
            self.models["clip_vit_b32"], self.processors["clip_vit_b32"] = clip.load(
                "ViT-B/32", device=self.device
            )
            print("✓ OpenAI CLIP ViT-B/32 loaded")
        except Exception as e:
            print(f"✗ Failed to load OpenAI CLIP ViT-B/32: {e}")

        # 2) OpenAI CLIP ViT-L/14
        try:
            print("Loading OpenAI CLIP ViT-L/14...")
            self.models["clip_vit_l14"], self.processors["clip_vit_l14"] = clip.load(
                "ViT-L/14", device=self.device
            )
            print("✓ OpenAI CLIP ViT-L/14 loaded")
        except Exception as e:
            print(f"✗ Failed to load OpenAI CLIP ViT-L/14: {e}")

        # 3) OpenCLIP ViT-B-32
        try:
            print("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
            self.models["openclip_vit_b32"], _, self.processors["openclip_vit_b32"] = (
                open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
                )
            )
            self.tokenizer_openclip = open_clip.get_tokenizer("ViT-B-32")
            print("✓ OpenCLIP ViT-B-32 loaded")
        except Exception as e:
            print(f"✗ Failed to load OpenCLIP: {e}")

        # 4) BLIP
        try:
            print("Loading BLIP model...")
            self.processors["blip"] = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.models["blip"] = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            print("✓ BLIP model loaded")
        except Exception as e:
            print(f"✗ Failed to load BLIP: {e}")

        # 5) DETR
        try:
            print("Loading DETR object detection model...")
            self.processors["detr"] = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self.models["detr"] = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50"
            ).to(self.device)
            print("✓ DETR model loaded")
        except Exception as e:
            print(f"✗ Failed to load DETR: {e}")

        print(f"Loaded {len(self.models)} models successfully!")

    # ---------- image utils ----------

    def load_image(self, image_url_or_pil):
        """Load image from URL or accept PIL.Image.Image"""
        try:
            if isinstance(image_url_or_pil, Image.Image):
                return image_url_or_pil.convert("RGB")

            image_url = str(image_url_or_pil).strip()
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            resp = requests.get(image_url, timeout=20, headers=headers)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {str(e)[:100]}")
            return None

    def detect_objects(self, image, confidence_threshold=0.7):
        """Detect objects using DETR"""
        try:
            if "detr" not in self.models:
                return []

            inputs = self.processors["detr"](images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.models["detr"](**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processors["detr"].post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            detected = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                box = [round(i, 2) for i in box.tolist()]
                label_name = self.models["detr"].config.id2label[label.item()]
                detected.append(
                    {
                        "label": label_name,
                        "confidence": round(score.item(), 3),
                        "box": box,
                    }
                )
            return detected
        except Exception as e:
            print(f"Error during object detection: {str(e)[:100]}")
            return []

    def crop_object_from_image(self, image, box, padding=10):
        try:
            x_min, y_min, x_max, y_max = box
            w, h = image.size
            x_min = max(0, int(x_min - padding))
            y_min = max(0, int(y_min - padding))
            x_max = min(w, int(x_max + padding))
            y_max = min(h, int(y_max + padding))
            return image.crop((x_min, y_min, x_max, y_max))
        except Exception as e:
            print(f"Error cropping image: {str(e)[:100]}")
            return None

    # ---------- classifiers ----------

    def classify_with_clip_openai(self, image, model_name, pickup_classes):
        try:
            if model_name not in self.models:
                return None
            model = self.models[model_name]
            preprocess = self.processors[model_name]
            im = preprocess(image).unsqueeze(0).to(self.device)
            txt = clip.tokenize(pickup_classes).to(self.device)
            with torch.no_grad():
                logits_per_image, _ = model(im, txt)
                probs = F.softmax(logits_per_image, dim=-1).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"Error in {model_name}: {str(e)[:80]}")
            return None

    def classify_with_openclip(self, image, pickup_classes):
        try:
            if "openclip_vit_b32" not in self.models or self.tokenizer_openclip is None:
                return None
            model = self.models["openclip_vit_b32"]
            preprocess = self.processors["openclip_vit_b32"]
            im = preprocess(image).unsqueeze(0).to(self.device)
            txt = self.tokenizer_openclip(pickup_classes).to(self.device)
            with torch.no_grad():
                img_f = model.encode_image(im)
                txt_f = model.encode_text(txt)
                img_f = F.normalize(img_f, dim=-1)
                txt_f = F.normalize(txt_f, dim=-1)
                logits = (img_f @ txt_f.T) * model.logit_scale.exp()
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            return probs
        except Exception as e:
            print(f"Error in OpenCLIP: {str(e)[:80]}")
            return None

    def get_blip_description(self, image):
        try:
            if "blip" not in self.models:
                return ""
            processor = self.processors["blip"]
            model = self.models["blip"]
            inputs = processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50)
            return processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in BLIP: {str(e)[:80]}")
            return ""

    def classify_with_blip_enhanced(self, image, pickup_classes):
        try:
            desc = self.get_blip_description(image)
            if not desc:
                return None
            dlow = desc.lower()
            scores = []
            for class_text in pickup_classes:
                # very light keywording
                words = (
                    class_text.replace("a photo of", "")
                    .replace("an", "")
                    .replace("a", "")
                    .strip()
                    .split()
                )
                key_terms = [w for w in words if len(w) > 3]
                score = sum(1 for t in key_terms if t.lower() in dlow) / max(
                    len(key_terms), 1
                )
                scores.append(score)
            scores = np.array(scores, dtype=np.float32)
            if scores.max() > 0:
                scores = scores / scores.sum()
            return scores
        except Exception as e:
            print(f"Error in BLIP-enhanced classification: {str(e)[:80]}")
            return None

    def ensemble_classify(self, image, pickup_classes):
        """
        Return a tuple (best_class: str, confidence: float, all_predictions: dict)
        """
        all_predictions = {}
        valid = []

        if "clip_vit_b32" in self.models:
            p = self.classify_with_clip_openai(image, "clip_vit_b32", pickup_classes)
            if p is not None:
                all_predictions["clip_vit_b32"] = p
                valid.append(p)

        if "clip_vit_l14" in self.models:
            p = self.classify_with_clip_openai(image, "clip_vit_l14", pickup_classes)
            if p is not None:
                all_predictions["clip_vit_l14"] = p
                valid.extend([p, p])  # double-weight

        if "openclip_vit_b32" in self.models:
            p = self.classify_with_openclip(image, pickup_classes)
            if p is not None:
                all_predictions["openclip_vit_b32"] = p
                valid.append(p)

        if "blip" in self.models:
            p = self.classify_with_blip_enhanced(image, pickup_classes)
            if p is not None:
                all_predictions["blip_enhanced"] = p
                valid.append(p * 0.5)  # light weight

        if not valid:
            return None, 0.0, all_predictions

        ensemble_probs = np.mean(valid, axis=0)
        best_idx = int(np.argmax(ensemble_probs))
        best_class = pickup_classes[best_idx]
        confidence = float(ensemble_probs[best_idx])
        return best_class, confidence, all_predictions

    def filter_relevant_objects(self, detected_objects):
        relevant_keywords = [
            "refrigerator",
            "laptop",
            "computer",
            "monitor",
            "tv",
            "television",
            "keyboard",
            "mouse",
            "printer",
            "scanner",
            "microwave",
            "washing machine",
            "fan",
            "chair",
            "scooter",
            "air conditioner",
            "battery",
            "book",
            "bottle",
            "metal",
            "plastic",
            "glass",
            "paper",
            "electronic",
            "appliance",
            "wire",
            "cable",
        ]
        return [
            obj
            for obj in detected_objects
            if any(k in obj["label"].lower() for k in relevant_keywords)
        ]

    def process_image(self, image_or_url, pickup_classes):
        """
        Complete pipeline: load, detect objects, classify relevant objects.
        Returns: (predicted_class, confidence, model_predictions, blip_description)
        """
        detection_threshold = 0.8  # Fixed threshold value
        image = self.load_image(image_or_url)
        if image is None:
            return None, 0.0, {}, ""

        blip_description = (
            self.get_blip_description(image) if "blip" in self.models else ""
        )

        detected_objects = self.detect_objects(
            image, confidence_threshold=detection_threshold
        )
        if not detected_objects:
            # classify whole image
            predicted_class, confidence, model_predictions = self.ensemble_classify(
                image, pickup_classes
            )
            return predicted_class, confidence, model_predictions, blip_description

        relevant_objects = self.filter_relevant_objects(detected_objects)
        if not relevant_objects:
            predicted_class, confidence, model_predictions = self.ensemble_classify(
                image, pickup_classes
            )
            return predicted_class, confidence, model_predictions, blip_description

        # classify each relevant object
        classification_results = []
        for obj in relevant_objects:
            cropped = self.crop_object_from_image(image, obj["box"])
            if cropped is None:
                continue
            pred, conf, preds = self.ensemble_classify(cropped, pickup_classes)
            if pred:
                classification_results.append(
                    {
                        "detected_object": obj["label"],
                        "detection_confidence": obj["confidence"],
                        "predicted_class": pred,
                        "classification_confidence": conf,
                        "bounding_box": obj["box"],
                        "model_predictions": preds,
                    }
                )

        if classification_results:
            best = max(
                classification_results, key=lambda x: x["classification_confidence"]
            )
            return (
                best["predicted_class"],
                float(best["classification_confidence"]),
                best["model_predictions"],
                blip_description,
            )

        # fallback to whole image
        predicted_class, confidence, model_predictions = self.ensemble_classify(
            image, pickup_classes
        )
        return predicted_class, confidence, model_predictions, blip_description


# ---------- STATIC CLASSES (your list) ----------
PICKUP_CLASSES = [
    "a photo of an air conditioner",
    "a photo of an outdoor air conditioner unit",
    "a photo of air conditioner grill scrap",
    "a photo of a split air conditioner",
    "a photo of a window air conditioner",
    "a photo of aluminum sheet scrap",
    "a photo of aluminum utensils scrap",
    "a photo of aluminum scrap",
    "a photo of aluminum parts",
    "a photo of a used inverter battery",
    "a photo of a dry battery",
    "a photo of a bicycle",
    "a photo of books",
    "a photo of brass scrap",
    "a photo of a CD case",
    "a photo of a computer CPU",
    "a photo of a CRT monitor",
    "a photo of cardboard boxes",
    "a photo of duplex cardboard",
    "a photo of raw cardboard",
    "a photo of a ceiling fan",
    "a photo of a plastic chair",
    "a photo of clothes",
    "a photo of iron scrap",
    "a photo of a set-top box",
    "a photo of air conditioner compressor scrap",
    "a photo of copper wires",
    "a photo of a desktop computer",
    "a photo of a DVD player or set-top box",
    "a photo of a deep freezer",
    "a photo of a double-door refrigerator",
    "a photo of a refrigerator door",
    "a photo of electronic e-waste",
    "a photo of a heating filament or element",
    "a photo of a refrigerator",
    "a photo of a front-load washing machine",
    "a photo of a fully automatic front-load washing machine",
    "a photo of a fully automatic top-load washing machine",
    "a photo of a copper water heater",
    "a photo of an iron or steel geyser",
    "a photo of broken glass",
    "a photo of glass bottles",
    "a photo of an inverter battery",
    "a photo of an electric iron",
    "a photo of a laptop charger",
    "a phot of charger or adapter",
    "a photo of a metal cooler",
    "a photo of an exhaust fan",
    "a photo of an LCD or LED screen",
    "a photo of a laptop",
    "a photo of a microwave oven",
    "a photo of mixed paper waste",
    "a photo of assorted plastic items",
    "a photo of mixed iron items",
    "a photo of aluminum-dominant scrap (90%)",
    "a photo of brass-dominant scrap (90%)",
    "a photo of iron-dominant scrap (90%)",
    "a photo of steel-dominant scrap (90%)",
    "a photo of a mixer or grinder",
    "a photo of a touchscreen mobile phone",
    "a photo of a TV or monitor",
    "a photo of an electric motor",
    "a photo of a fan motor or wall fan",
    "a photo of old newspapers",
    "a photo of a mobile phone",
    "a photo of assorted plastic",
    "a photo of a plastic air cooler",
    "a photo of household plastic appliances",
    "a photo of a plastic fan",
    "a photo of a plastic sheet",
    "a photo of a printer or scanner",
    "a photo of office electronics (fax, printer, LCD, LED)",
    "a photo of an RO water purifier",
    "a photo of a router or modem",
    "a photo of a scooter",
    "a photo of a scooty",
    "a photo of a semi-automatic washing machine",
    "a photo of a single-door refrigerator",
    "a photo of small iron appliances",
    "a photo of small plastic appliances",
    "a photo of a power stabilizer",
    "a photo of an inverter",
    "a photo of a UPS unit",
    "a photo of stainless steel",
    "a photo of steel grade 304",
    "a photo of a television",
    "a photo of a tablet device",
    "a photo of test electronics",
    "a photo of tin containers",
    "a photo of large metal junk",
    "a photo of medium metal junk",
    "a photo of small metal junk",
    "a photo of a top-load washing machine",
    "a photo of a treadmill",
    "a photo of scrap tires",
    "a photo of electrical wires",
    "a photo of fitting wire scrap",
    "a photo of mixed wire scrap",
    "a photo of raw aluminum",
    "a photo of a toy car",
    "a photo of mixed brass and steel scrap",
    "a photo of a kitchen chimney hood",
    "a photo of a traditional stove scrap",
    "a photo of worn-out clothes",
    "a photo of an old computer cabinet",
    "a photo of torn paper pieces",
    "a photo of a double-drum washing machine",
    "a photo of a DVD case",
    "a photo of a refrigerator",
    "a photo of a fridge with damaged compressor",
    "a photo of an old fridge",
    "a photo of a damaged refrigerator",
    "a photo of a front-load washing machine",
    "a photo of a broken refrigerator door",
    "a photo of cracked glass",
    "a photo of a broken glass bottle",
    "a photo of miscellaneous items",
    "a photo of a broken cooler",
    "a photo of computer waste (motherboard, CPU, power supply)",
    "a photo of a computer keyboard",
    "a photo of metal household appliances",
    "a photo of small metal appliances",
    "a photo of a compact car",
    "a photo of a mini refrigerator",
    "a photo of a computer monitor",
    "a photo of a smartphone",
    "a photo of an unknown object (Pilthoni)",
    "a photo of plastic e-waste",
    "a photo of a plastic remote control",
    "a photo of broken plastic buttons",
    "a photo of a plastic speaker",
    "a photo of a plastic toy",
    "a photo of a general refrigerator",
    "a photo of an RO pump or motor",
    "a photo of a broken solar panel",
    "a phot of polythene bags",
    "a phot of polythene bag of scrap",
    "a photo of a power bank",
    "a photo of a sack of scrap",
    "a photo of a speaker box",
    "a photo of a Bluetooth speaker",
    "a photo of a sound system",
    "a photo of a split air conditioner",
    "a photo of a 1.5-ton Split AC",
    "a photo of a split air conditioning unit",
    "a photo of an old stabilizer",
    "a photo of steel scrap",
    "a photo of broken steel items",
    "a photo of tin scrap",
    "a photo of a traditional stove (Chulha) scrap",
    "a photo of a lathe or trade machine",
    "a photo of a commercial washing machine",
    "a photo of a water cooler",
    "a photo of air cooler",
    "a photo of an X-ray machine",
]


# ---------- STATIC CLASSES (your list) ----------
PICKUP_CLASSES = [
    "a photo of an air conditioner",
    "a photo of an outdoor air conditioner unit",
    "a photo of air conditioner grill scrap",
    "a photo of a split air conditioner",
    "a photo of a window air conditioner",
    "a photo of aluminum sheet scrap",
    "a photo of aluminum utensils scrap",
    "a photo of aluminum scrap",
    "a photo of aluminum parts",
    "a photo of a used inverter battery",
    "a photo of a dry battery",
    "a photo of a bicycle",
    "a photo of books",
    "a photo of brass scrap",
    "a photo of a CD case",
    "a photo of a computer CPU",
    "a photo of a CRT monitor",
    "a photo of cardboard boxes",
    "a photo of duplex cardboard",
    "a photo of raw cardboard",
    "a photo of a ceiling fan",
    "a photo of a plastic chair",
    "a photo of clothes",
    "a photo of iron scrap",
    "a photo of a set-top box",
    "a photo of air conditioner compressor scrap",
    "a photo of copper wires",
    "a photo of a desktop computer",
    "a photo of a DVD player or set-top box",
    "a photo of a deep freezer",
    "a photo of a double-door refrigerator",
    "a photo of a refrigerator door",
    "a photo of electronic e-waste",
    "a photo of a heating filament or element",
    "a photo of a refrigerator",
    "a photo of a front-load washing machine",
    "a photo of a fully automatic front-load washing machine",
    "a photo of a fully automatic top-load washing machine",
    "a photo of a copper water heater",
    "a photo of an iron or steel geyser",
    "a photo of broken glass",
    "a photo of glass bottles",
    "a photo of an inverter battery",
    "a photo of an electric iron",
    "a photo of a laptop charger",
    "a phot of charger or adapter",
    "a photo of a metal cooler",
    "a photo of an exhaust fan",
    "a photo of an LCD or LED screen",
    "a photo of a laptop",
    "a photo of a microwave oven",
    "a photo of mixed paper waste",
    "a photo of assorted plastic items",
    "a photo of mixed iron items",
    "a photo of aluminum-dominant scrap (90%)",
    "a photo of brass-dominant scrap (90%)",
    "a photo of iron-dominant scrap (90%)",
    "a photo of steel-dominant scrap (90%)",
    "a photo of a mixer or grinder",
    "a photo of a touchscreen mobile phone",
    "a photo of a TV or monitor",
    "a photo of an electric motor",
    "a photo of a fan motor or wall fan",
    "a photo of old newspapers",
    "a photo of a mobile phone",
    "a photo of assorted plastic",
    "a photo of a plastic air cooler",
    "a photo of household plastic appliances",
    "a photo of a plastic fan",
    "a photo of a plastic sheet",
    "a photo of a printer or scanner",
    "a photo of office electronics (fax, printer, LCD, LED)",
    "a photo of an RO water purifier",
    "a photo of a router or modem",
    "a photo of a scooter",
    "a photo of a scooty",
    "a photo of a semi-automatic washing machine",
    "a photo of a single-door refrigerator",
    "a photo of small iron appliances",
    "a photo of small plastic appliances",
    "a photo of a power stabilizer",
    "a photo of an inverter",
    "a photo of a UPS unit",
    "a photo of stainless steel",
    "a photo of steel grade 304",
    "a photo of a television",
    "a photo of a tablet device",
    "a photo of test electronics",
    "a photo of tin containers",
    "a photo of large metal junk",
    "a photo of medium metal junk",
    "a photo of small metal junk",
    "a photo of a top-load washing machine",
    "a photo of a treadmill",
    "a photo of scrap tires",
    "a photo of electrical wires",
    "a photo of fitting wire scrap",
    "a photo of mixed wire scrap",
    "a photo of raw aluminum",
    "a photo of a toy car",
    "a photo of mixed brass and steel scrap",
    "a photo of a kitchen chimney hood",
    "a photo of a traditional stove scrap",
    "a photo of worn-out clothes",
    "a photo of an old computer cabinet",
    "a photo of torn paper pieces",
    "a photo of a double-drum washing machine",
    "a photo of a DVD case",
    "a photo of a refrigerator",
    "a photo of a fridge with damaged compressor",
    "a photo of an old fridge",
    "a photo of a damaged refrigerator",
    "a photo of a front-load washing machine",
    "a photo of a broken refrigerator door",
    "a photo of cracked glass",
    "a photo of a broken glass bottle",
    "a photo of miscellaneous items",
    "a photo of a broken cooler",
    "a photo of computer waste (motherboard, CPU, power supply)",
    "a photo of a computer keyboard",
    "a photo of metal household appliances",
    "a photo of small metal appliances",
    "a photo of a compact car",
    "a photo of a mini refrigerator",
    "a photo of a computer monitor",
    "a photo of a smartphone",
    "a photo of an unknown object (Pilthoni)",
    "a photo of plastic e-waste",
    "a photo of a plastic remote control",
    "a photo of broken plastic buttons",
    "a photo of a plastic speaker",
    "a photo of a plastic toy",
    "a photo of a general refrigerator",
    "a photo of an RO pump or motor",
    "a photo of a broken solar panel",
    "a phot of polythene bags",
    "a phot of polythene bag of scrap",
    "a photo of a power bank",
    "a photo of a sack of scrap",
    "a photo of a speaker box",
    "a photo of a Bluetooth speaker",
    "a photo of a sound system",
    "a photo of a split air conditioner",
    "a photo of a 1.5-ton Split AC",
    "a photo of a split air conditioning unit",
    "a photo of an old stabilizer",
    "a photo of steel scrap",
    "a photo of broken steel items",
    "a photo of tin scrap",
    "a photo of a traditional stove (Chulha) scrap",
    "a photo of a lathe or trade machine",
    "a photo of a commercial washing machine",
    "a photo of a water cooler",
    "a photo of air cooler",
    "a photo of an X-ray machine",
]


# .venv\Scripts\activate

# activate it first
# uvicorn main:app --reload