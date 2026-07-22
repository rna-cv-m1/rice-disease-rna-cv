"""
Module pour le chargement et la gestion des modèles
======================================================

Gère :
- Chargement des modèles pré-entraînés
- Cache des modèles pour performance
- Validation des poids
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Chemin vers le dossier models
MODELS_DIR = Path(__file__).parent.parent / "models"

class ModelManager:
    """Gestionnaire des modèles**"""
    
    _models_cache = {}  # Cache pour éviter rechargements
    
    AVAILABLE_MODELS = {
        "ResNet50": "resnet50.pth",
        "VGG16": "vgg16.pth",
        "EfficientNet": "efficientnet.pth",
    }
    
    @classmethod
    def list_available_models(cls) -> list:
        """Lister les modèles disponibles"""
        available = []
        for name, filename in cls.AVAILABLE_MODELS.items():
            path = MODELS_DIR / filename
            if path.exists():
                available.append(name)
        return available
    
    @classmethod
    def load_model(cls, model_name: str, device: str = "cpu"):
        """
        Charger un modèle pré-entraîné
        
        Args:
            model_name: Nom du modèle ("ResNet50", "VGG16", "EfficientNet")
            device: Appareil pour le calcul ("cpu" ou "cuda")
        
        Returns:
            Modèle chargé
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        # Vérifier le cache
        cache_key = f"{model_name}_{device}"
        if cache_key in cls._models_cache:
            logger.info(f"Modèle {model_name} chargé depuis cache")
            return cls._models_cache[cache_key]
        
        # Charger depuis fichier
        model_path = MODELS_DIR / cls.AVAILABLE_MODELS[model_name]
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fichier modèle non trouvé : {model_path}\n"
                f"Modèles disponibles : {cls.list_available_models()}"
            )
        
        try:
            # Import lazy pour éviter dépendances inutiles
            import torch
            
            model = torch.load(str(model_path), map_location=device)
            model.eval()  # Mode évaluation
            
            # Mettre en cache
            cls._models_cache[cache_key] = model
            
            logger.info(f"Modèle {model_name} chargé depuis {model_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Obtenir les informations d'un modèle"""
        model_path = MODELS_DIR / cls.AVAILABLE_MODELS[model_name]
        
        return {
            "name": model_name,
            "path": str(model_path),
            "file_exists": model_path.exists(),
            "file_size_mb": model_path.stat().st_size / (1024**2) if model_path.exists() else None,
        }

    @classmethod
    def clear_cache(cls):
        """Vider le cache des modèles"""
        cls._models_cache.clear()
        logger.info("Cache des modèles vidé")
