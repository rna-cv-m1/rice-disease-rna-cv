#!/usr/bin/env python
"""
Script de démonstration de l'interface
=======================================

Teste les composants principaux avant de lancer l'interface Streamlit.
"""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_image_preprocessor():
    """Démonstrer ImagePreprocessor"""
    print("\n" + "="*60)
    print("📊 TEST 1: ImagePreprocessor")
    print("="*60)
    
    from src.utils import ImagePreprocessor
    from PIL import Image
    
    # Créer une image de test
    test_image = Image.new('RGB', (400, 400), color='red')
    
    print(f"Image créée : {test_image.size} pixels")
    
    # Prétraiter
    processed = ImagePreprocessor.preprocess(test_image)
    print(f"Image prétraitée : shape {processed.shape}")
    print(f"  - Valeur min: {processed.min():.4f}")
    print(f"  - Valeur max: {processed.max():.4f}")
    
    # Info image
    info = ImagePreprocessor.get_image_info(test_image)
    print(f"Metadata : {info}")
    
    return True

def demo_model_manager():
    """Démonstrer ModelManager"""
    print("\n" + "="*60)
    print("🤖 TEST 2: ModelManager")
    print("="*60)
    
    from src.model_manager import ModelManager
    
    # Lister modèles disponibles
    available = ModelManager.list_available_models()
    print(f"Modèles disponibles : {available}")
    
    if not available:
        print("⚠️  Aucun modèle trouvé dans le dossier models/")
        print("   → À implémenter : ajouter les fichiers .pth")
        return False
    
    # Info modèles
    for model_name in ["ResNet50", "VGG16", "EfficientNet"]:
        if model_name in available:
            info = ModelManager.get_model_info(model_name)
            print(f"\n{model_name}:")
            print(f"  Chemin : {info['path']}")
            print(f"  Taille : {info['file_size_mb']:.2f} MB")
    
    return True

def demo_prediction_postprocessor():
    """Démonstrer PredictionPostprocessor"""
    print("\n" + "="*60)
    print("📈 TEST 3: PredictionPostprocessor")
    print("="*60)
    
    from src.utils import PredictionPostprocessor
    import numpy as np
    
    # Simuler aortie du modèle
    logits = np.array([2.1, 0.5, -0.5, -1.2, -0.1])
    
    print(f"Logits simulés : {logits}")
    
    # Traiter
    predictions = PredictionPostprocessor.process_predictions(logits)
    print(f"\nPrédictions (après softmax) :")
    for disease, prob in predictions.items():
        print(f"  {disease}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Top prediction
    top_class, top_prob = PredictionPostprocessor.get_top_prediction(predictions)
    print(f"\nMeilleure prédiction : {top_class} ({top_prob*100:.1f}%)")
    
    # DataFrame
    try:
        df = PredictionPostprocessor.get_prediction_dataframe(predictions)
        print(f"\nDataFrame pandas :")
        print(df.to_string(index=False))
    except ImportError:
        print("⚠️  pandas non installé")
    
    return True

def demo_validation_utils():
    """Démonstrer ValidationUtils"""
    print("\n" + "="*60)
    print("✅ TEST 4: ValidationUtils")
    print("="*60)
    
    from src.utils import ValidationUtils
    from PIL import Image
    
    # Formats supportés
    print(f"Formats supportés : {ValidationUtils.SUPPORTED_FORMATS}")
    print(f"Taille max : {ValidationUtils.MAX_IMAGE_SIZE_MB} MB")
    print(f"Taille min : {ValidationUtils.MIN_IMAGE_SIZE_PX} px")
    
    # Créer une image de test
    test_image = Image.new('RGB', (224, 224))
    test_path = Path("test_image.png")
    test_image.save(test_path)
    
    # Valider
    is_valid, msg = ValidationUtils.validate_image_file(str(test_path))
    print(f"\nValidation : {msg}")
    
    # Warnings qualité
    warnings = ValidationUtils.get_quality_warning(test_image)
    print(f"Warnings qualité : {warnings if warnings else 'Aucun'}")
    
    # Cleanup
    test_path.unlink()
    
    return True

def main():
    """Exécuter tous les tests"""
    print("\n" + "🌾 "*20)
    print("DÉMONSTRATION - Interface Détection Maladies du Riz")
    print("🌾 "*20)
    
    tests = [
        ("ImagePreprocessor", demo_image_preprocessor),
        ("ModelManager", demo_model_manager),
        ("PredictionPostprocessor", demo_prediction_postprocessor),
        ("ValidationUtils", demo_validation_utils),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ ERREUR dans {test_name} : {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("📋 RÉSUMÉ")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} : {test_name}")
    
    total_pass = sum(results.values())
    total = len(results)
    print(f"\nTotal : {total_pass}/{total} tests réussis")
    
    if total_pass == total:
        print("\n✅ Tous les tests sont passés !")
        print("L'interface est prête à être lancée :")
        print("\n  streamlit run src/app.py\n")
        return 0
    else:
        print(f"\n⚠️  {total - total_pass} test(s) échoué(s)")
        print("À corriger avant de lancer l'interface...")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
