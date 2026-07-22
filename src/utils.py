"""
Module utilitaire pour l'interface
===================================

Contient :
- Prétraitement des images
- Posttraitement des prédictions
- Utilitaires généraux
"""

from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Classes de maladies supportées
DISEASE_CLASSES = {
    0: "Saine",
    1: "Cercosporiose",
    2: "Pyriculariose",
    3: "Brûlure bactérienne",
    4: "Tache brune",
}

CLASS_COLORS = {
    "Saine": "#2ecc71",
    "Cercosporiose": "#f39c12",
    "Pyriculariose": "#e74c3c",
    "Brûlure bactérienne": "#f39c12",
    "Tache brune": "#e67e22",
}

class ImagePreprocessor:
    """Prétraitement des images pour la prédiction"""
    
    DEFAULT_SIZE = (224, 224)
    
    @staticmethod
    def preprocess(image: Image.Image, target_size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
        """
        Prétraiter une image PIL pour la prédiction
        
        Args:
            image: Image PIL ouverte
            target_size: Taille cible (par défaut 224x224)
        
        Returns:
            Array numpy preprocessé et normalisé
        """
        # Redimensionner
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convertir RGB si nécessaire
        if image_resized.mode != "RGB":
            image_resized = image_resized.convert("RGB")
        
        # Convertir en array numpy [0, 1]
        image_array = np.array(image_resized) / 255.0
        
        # Normalisation ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_array - mean) / std
        
        # Ajouter dimension batch et convertir en CHW
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0)
        
        return image_batch.astype(np.float32)
    
    @staticmethod
    def get_image_info(image: Image.Image) -> Dict:
        """Obtenir les informations sur une image"""
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "width": image.width,
            "height": image.height,
        }


class PredictionPostprocessor:
    """Posttraitement des prédictions du modèle"""
    
    @staticmethod
    def process_predictions(
        predictions: np.ndarray,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Convertir les outputs du modèle en prédictions lisibles
        
        Args:
            predictions: Output du modèle (logits ou probabilités)
            threshold: Seuil minimum (0 = accepter tous)
        
        Returns:
            Dict avec classes et probabilités
        """
        # Appliquer softmax si nécessaire (si logits)
        if predictions.max() > 1.0 or predictions.min() < 0:
            predictions = PredictionPostprocessor._softmax(predictions)
        
        # Créer dict classe -> probabilité
        result = {}
        for class_id, class_name in DISEASE_CLASSES.items():
            prob = float(predictions[class_id])
            if prob >= threshold:
                result[class_name] = prob
        
        return result
    
    @staticmethod
    def get_top_prediction(predictions: Dict[str, float]) -> Tuple[str, float]:
        """Obtenir la prédiction avec la probabilité la plus élevée"""
        if not predictions:
            return "Indéterminé", 0.0
        
        top_class = max(predictions, key=predictions.get)
        top_prob = predictions[top_class]
        return top_class, top_prob
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Calculer softmax"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    @staticmethod
    def get_prediction_dataframe(predictions: Dict[str, float]):
        """Convertir prédictions en DataFrame"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(
                [(k, v) for k, v in predictions.items()],
                columns=["Maladie", "Probabilité"]
            ).sort_values("Probabilité", ascending=False)
            
            return df
        except ImportError:
            logger.warning("pandas non installé, retour du dict brut")
            return predictions


class ValidationUtils:
    """Utilitaires de validation"""
    
    SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "bmp"}
    MAX_IMAGE_SIZE_MB = 200
    MIN_IMAGE_SIZE_PX = 100
    
    @staticmethod
    def validate_image_file(filepath: str) -> Tuple[bool, str]:
        """
        Valider un fichier image
        
        Args:
            filepath: Chemin du fichier
        
        Returns:
            (is_valid, message)
        """
        # Vérifier format
        ext = filepath.lower().split(".")[-1]
        if ext not in ValidationUtils.SUPPORTED_FORMATS:
            return False, f"Format non supporté : .{ext}"
        
        # Vérifier taille fichier
        try:
            import os
            size_mb = os.path.getsize(filepath) / (1024**2)
            if size_mb > ValidationUtils.MAX_IMAGE_SIZE_MB:
                return False, f"Image trop voluminense : {size_mb:.1f}MB (max: {ValidationUtils.MAX_IMAGE_SIZE_MB}MB)"
        except OSError:
            return False, "Impossible de lire le fichier"
        
        # Vérifier que c'est une image valide
        try:
            img = Image.open(filepath)
            img.verify()
            img = Image.open(filepath)  # Réouvrir après verify
            
            if img.size[0] < ValidationUtils.MIN_IMAGE_SIZE_PX or img.size[1] < ValidationUtils.MIN_IMAGE_SIZE_PX:
                return False, f"Image trop petite : {img.size} (min: {ValidationUtils.MIN_IMAGE_SIZE_PX}px)"
            
            return True, "Image valide"
        except Exception as e:
            return False, f"Image corrompue ou invalide : {str(e)}"
    
    @staticmethod
    def get_quality_warning(image: Image.Image) -> List[str]:
        """Obtenir les avertissements de qualité sur une image"""
        warnings = []
        
        # Résolution
        if image.size[0] < 224 or image.size[1] < 224:
            warnings.append("⚠️ Résolution faible (min conseillé: 224x224)")
        
        # Ratio d'aspect
        ratio = max(image.size) / min(image.size)
        if ratio > 2:
            warnings.append("⚠️ Ratio d'aspect anormal (image très étirée/aplatie)")
        
        return warnings
