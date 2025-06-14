"""
PIPELINE COMPLETA INCEPTION V4 - CLASSIFICAÇÃO ALN

Este conjunto de arquivos implementa uma pipeline completa para classificação 
de metástase de linfonodos axilares (ALN) usando o modelo Inception V4.

ARQUIVOS CRIADOS:
================

1. inception_pipeline.py
   - Classe principal InceptionALNClassifier
   - Preparação de dados, treinamento e avaliação
   - Modificação do modelo Inception V4 para 3 classes

2. train_inception.py  
   - Script de treinamento com diferentes modos
   - Interface de linha de comando
   - Controle de hiperparâmetros

3. inception_predictor.py
   - Predição de novos patches
   - Agregação de resultados por paciente
   - Visualização de resultados

4. compare_methods.py
   - Comparação entre Inception V4 e XGBoost
   - Análise estatística detalhada
   - Gráficos comparativos

COMO USAR:
=========

1. TREINAMENTO:
   python train_inception.py
   
   Opções:
   - Treinamento rápido (5 patches/paciente, 10 épocas)
   - Treinamento médio (10 patches/paciente, 30 épocas)  
   - Treinamento completo (todos patches, 50 épocas)

2. PREDIÇÃO:
   python inception_predictor.py
   
   Pode predizer:
   - Um paciente específico
   - Todos os pacientes em lote

3. COMPARAÇÃO:
   python compare_methods.py
   
   Compara performance entre:
   - Inception V4 (classificação de patches)
   - XGBoost (features morfológicos)

ESTRUTURA DE DADOS:
==================

Entrada esperada:
- patches/{patient_id}/{patch_files}.jpg
- patient-clinical-data.csv com coluna "ALN status"

Classes:
- N0: Sem metástase
- N+(1-2): 1-2 linfonodos positivos
- N+(>2): Mais de 2 linfonodos positivos

REQUISITOS:
===========

- TensorFlow/Keras
- OpenCV
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

EXEMPLO DE USO PROGRAMÁTICO:
===========================

from inception_pipeline import InceptionALNClassifier

# Treinar modelo
classifier = InceptionALNClassifier("patches", "patient-clinical-data.csv")
datasets = classifier.prepare_dataset(max_patches_per_patient=10)
model = classifier.build_model()
history = classifier.train_model(datasets, epochs=30)
results = classifier.evaluate_model(datasets['test'])

# Predizer novos dados
from inception_predictor import InceptionALNPredictor
predictor = InceptionALNPredictor("model.h5")
results = predictor.predict_patient_patches("patient_123_patches/")

CONFIGURAÇÕES RECOMENDADAS:
===========================

Para teste rápido:
- max_patches_per_patient=5
- epochs=10
- batch_size=8

Para treinamento final:
- max_patches_per_patient=None (todos)
- epochs=50
- batch_size=4 (para evitar problemas de memória)

OBSERVAÇÕES IMPORTANTES:
========================

1. O modelo Inception V4 original está em inception.py e NÃO foi modificado
2. A pipeline modifica apenas a camada final para 3 classes
3. Usa data augmentation durante treinamento
4. Implementa separação por paciente (não por patch) para evitar data leakage
5. Suporta agregação de predições por paciente (voto majoritário, média de probabilidades)
6. Inclui visualizações detalhadas e métricas de avaliação

RESULTADOS ESPERADOS:
====================

O modelo deve atingir:
- Acurácia > 70% na classificação de patches
- Melhor performance na classe N0 (mais amostras)
- Comparação interessante com XGBoost baseado em features morfológicos

ARQUIVOS GERADOS:
================

Durante execução, são gerados:
- best_inception_aln_model.h5 (melhor modelo durante treinamento)
- inception_v4_aln_final.h5 (modelo final)
- predictions.csv (predições em lote)
- method_comparison_results.csv (comparação de métodos)
- Gráficos de treinamento e avaliação

"""

# Função de verificação do ambiente
def check_environment():
    """Verifica se o ambiente está configurado corretamente"""
    
    import sys
    from pathlib import Path
    
    print("🔍 VERIFICAÇÃO DO AMBIENTE")
    print("="*40)
    
    # Verifica Python
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️ Recomendado Python 3.7+")
    
    # Verifica bibliotecas
    required_packages = [
        'tensorflow', 'opencv-python', 'scikit-learn',
        'matplotlib', 'seaborn', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                print(f"✅ TensorFlow: {tf.__version__}")
            elif package == 'scikit-learn':
                import sklearn
                print(f"✅ scikit-learn: {sklearn.__version__}")
            else:
                exec(f"import {package}")
                print(f"✅ {package}: instalado")
                
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: não encontrado")
    
    # Verifica GPU
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            print("✅ GPU: disponível")
        else:
            print("⚠️ GPU: não disponível (usando CPU)")
    except:
        pass
    
    # Verifica arquivos do projeto
    print(f"\n📁 ARQUIVOS DO PROJETO")
    print("="*40)
    
    base_dir = Path(__file__).parent
    
    required_files = [
        'inception.py',
        'patient-clinical-data.csv',
        'patches'
    ]
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            if full_path.is_dir():
                patch_count = len(list(full_path.iterdir()))
                print(f"✅ {file_path}: {patch_count} itens")
            else:
                print(f"✅ {file_path}: encontrado")
        else:
            print(f"❌ {file_path}: não encontrado")
    
    # Resumo
    print(f"\n📋 RESUMO")
    print("="*40)
    
    if missing_packages:
        print(f"❌ Pacotes faltantes: {', '.join(missing_packages)}")
        print(f"💡 Instale com: pip install {' '.join(missing_packages)}")
    else:
        print("✅ Todas as dependências estão instaladas")
        
    print(f"\n🚀 Para começar, execute:")
    print(f"   python train_inception.py")

if __name__ == "__main__":
    check_environment()
