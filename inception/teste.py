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
    
    if python_version < (3, 10):
        print("⚠️ Recomendado Python 3.10+")
    
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
    
    base_dir = Path(__file__).parent.parent
    
    required_files = [
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