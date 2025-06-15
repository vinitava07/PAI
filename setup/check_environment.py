#!/usr/bin/env python3
"""
Script para verificar se o ambiente está configurado corretamente
para executar o treinamento do Inception V3
"""

import sys
import os
from pathlib import Path
import importlib.util
import pandas as pd


def check_tensorflow():
    """Verifica instalação do TensorFlow"""
    print("\n🔍 Verificando TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} instalado")
        
        # Verifica GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU disponível: {len(gpus)} dispositivo(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  Nenhuma GPU encontrada - treinamento será mais lento")
        
        return True
    except ImportError:
        print("❌ TensorFlow não instalado!")
        print("   Execute: pip install tensorflow")
        return False


def check_dependencies():
    """Verifica todas as dependências"""
    print("\n🔍 Verificando dependências...")
    
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    all_ok = True
    for module, package in dependencies.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            print(f"✅ {package} instalado")
        else:
            print(f"❌ {package} não instalado!")
            print(f"   Execute: pip install {package}")
            all_ok = False
    
    return all_ok


def check_data_structure():
    """Verifica estrutura de dados"""
    print("\n🔍 Verificando estrutura de dados...")
    
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    csv_path = base_dir / "patient-clinical-data.csv"
    
    # Verifica CSV
    if not csv_path.exists():
        print(f"❌ Arquivo CSV não encontrado: {csv_path}")
        return False
    else:
        print(f"✅ Arquivo CSV encontrado: {csv_path}")
        
        # Tenta ler o CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"   - {len(df)} pacientes no CSV")
            
            # Verifica colunas necessárias
            required_cols = ['Patient ID', 'ALN status']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Colunas faltando no CSV: {missing_cols}")
                return False
            else:
                print("✅ Todas as colunas necessárias presentes")
            
            # Verifica valores de ALN status
            aln_values = df['ALN status'].value_counts()
            print("\n   Distribuição ALN status:")
            for status, count in aln_values.items():
                print(f"   - {status}: {count}")
            
        except Exception as e:
            print(f"❌ Erro ao ler CSV: {e}")
            return False
    
    # Verifica diretório de patches
    if not patches_dir.exists():
        print(f"\n❌ Diretório de patches não encontrado: {patches_dir}")
        return False
    else:
        print(f"\n✅ Diretório de patches encontrado: {patches_dir}")
        
        # Conta pastas de pacientes
        patient_dirs = [d for d in patches_dir.iterdir() if d.is_dir()]
        print(f"   - {len(patient_dirs)} pastas de pacientes")
        
        # Verifica algumas pastas aleatórias
        import random
        sample_dirs = random.sample(patient_dirs, min(5, len(patient_dirs)))
        
        print("\n   Verificando amostras:")
        for patient_dir in sample_dirs:
            patches = list(patient_dir.glob("*.jpg")) + list(patient_dir.glob("*.png"))
            print(f"   - Paciente {patient_dir.name}: {len(patches)} patches")
        
        # Verifica se há patches
        total_patches = sum(
            len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) 
            for d in patient_dirs
        )
        
        if total_patches == 0:
            print("\n❌ Nenhum patch encontrado!")
            return False
        else:
            print(f"\n✅ Total de patches: ~{total_patches} (estimado)")
    
    return True


def check_disk_space():
    """Verifica espaço em disco"""
    print("\n🔍 Verificando espaço em disco...")
    
    import shutil
    
    base_dir = Path(__file__).parent
    stat = shutil.disk_usage(base_dir)
    
    # Converte para GB
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_percent = (stat.used / stat.total) * 100
    
    print(f"   Espaço livre: {free_gb:.1f} GB")
    print(f"   Espaço total: {total_gb:.1f} GB")
    print(f"   Uso: {used_percent:.1f}%")
    
    if free_gb < 5:
        print("⚠️  Pouco espaço em disco! Recomendado pelo menos 5GB livres")
        return False
    else:
        print("✅ Espaço em disco suficiente")
        return True


def check_memory():
    """Verifica memória RAM"""
    print("\n🔍 Verificando memória...")
    
    try:
        import psutil
        
        # Memória total e disponível
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"   RAM total: {total_gb:.1f} GB")
        print(f"   RAM disponível: {available_gb:.1f} GB")
        
        if total_gb < 8:
            print("⚠️  Pouca RAM! Recomendado pelo menos 8GB")
            print("   Considere usar batch_size menor")
        else:
            print("✅ Memória RAM suficiente")
        
        return True
        
    except ImportError:
        print("⚠️  psutil não instalado - não foi possível verificar memória")
        return True


def main():
    """Executa todas as verificações"""
    print("="*60)
    print("🏥 VERIFICAÇÃO DO AMBIENTE - INCEPTION V3 ALN")
    print("="*60)
    
    checks = [
        ("TensorFlow", check_tensorflow),
        ("Dependências", check_dependencies),
        ("Estrutura de dados", check_data_structure),
        ("Espaço em disco", check_disk_space),
        ("Memória", check_memory)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro ao verificar {name}: {e}")
            results.append((name, False))
    
    # Resumo
    print("\n" + "="*60)
    print("📋 RESUMO DAS VERIFICAÇÕES")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ OK" if passed else "❌ FALHOU"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ Ambiente pronto para treinamento!")
        print("\nPróximo passo:")
        print("  python run_inception_training.py")
    else:
        print("\n❌ Corrija os problemas acima antes de iniciar o treinamento")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
