import os
import sys
from pathlib import Path
import tensorflow as tf

# Configura GPU se disponível
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU disponível: {physical_devices[0]}")
else:
    print("Executando em CPU")

from inception_pipeline import InceptionALNClassifier

def run_inception_training(max_patches_per_patient=10, epochs=50, batch_size=8):
    """
    Executa treinamento completo do Inception V4
    
    Args:
        max_patches_per_patient: Máximo de patches por paciente
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
    """
    
    print("="*70)
    print("🧠 TREINAMENTO INCEPTION V4 - CLASSIFICAÇÃO ALN")
    print("="*70)
    
    # Configuração de caminhos
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    # Verifica se arquivos existem
    if not patches_dir.exists():
        print(f"❌ Diretório de patches não encontrado: {patches_dir}")
        return None
        
    if not clinical_data_path.exists():
        print(f"❌ Arquivo CSV não encontrado: {clinical_data_path}")
        return None
    
    print(f"📁 Patches: {patches_dir}")
    print(f"📋 Dados clínicos: {clinical_data_path}")
    print(f"🖼️ Max patches por paciente: {max_patches_per_patient}")
    print(f"🔄 Épocas: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    
    try:
        # Inicializa classificador
        classifier = InceptionALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Prepara dataset
        print("\n" + "="*50)
        print("📊 PREPARAÇÃO DOS DADOS")
        print("="*50)
        
        datasets = classifier.prepare_dataset(
            max_patches_per_patient=max_patches_per_patient,
            test_size=0.2,
            val_size=0.1
        )
        
        # Constrói modelo
        print("\n" + "="*50)
        print("🏗️ CONSTRUÇÃO DO MODELO")
        print("="*50)
        
        model = classifier.build_model()
        
        # Treina modelo
        print("\n" + "="*50)
        print("🚀 TREINAMENTO")
        print("="*50)
        
        history = classifier.train_model(
            datasets=datasets,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Avalia modelo
        print("\n" + "="*50)
        print("📈 AVALIAÇÃO")
        print("="*50)
        
        results = classifier.evaluate_model(
            test_df=datasets['test'],
            batch_size=batch_size
        )
        
        # Visualizações
        print("\n" + "="*50)
        print("📊 VISUALIZAÇÕES")
        print("="*50)
        
        classifier.plot_training_history(history)
        classifier.plot_confusion_matrix(results['confusion_matrix'])
        
        # Salva modelo
        model_path = base_dir / "inception_v4_aln_final.h5"
        classifier.save_model(str(model_path))
        
        # Relatório final
        print("\n" + "="*50)
        print("📋 RELATÓRIO FINAL")
        print("="*50)
        
        print(f"✅ Treinamento concluído com sucesso!")
        print(f"🎯 Acurácia final: {results['accuracy']:.4f}")
        print(f"💾 Modelo salvo em: {model_path}")
        
        # Métricas por classe
        report = results['classification_report']
        print("\n📊 Métricas por classe:")
        for class_name in classifier.class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
        
        # Estatísticas do dataset
        print(f"\n📈 Estatísticas do dataset:")
        print(f"  Total de patches: {len(datasets['all'])}")
        print(f"  Patches treino: {len(datasets['train'])}")
        print(f"  Patches validação: {len(datasets['val'])}")
        print(f"  Patches teste: {len(datasets['test'])}")
        
        return classifier, history, results
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_inception_evaluation(model_path, max_patches_per_patient=None):
    """
    Executa apenas avaliação com modelo pré-treinado
    
    Args:
        model_path: Caminho do modelo salvo
        max_patches_per_patient: Máximo de patches por paciente
    """
    
    print("="*70)
    print("🔍 AVALIAÇÃO INCEPTION V4 - MODELO PRÉ-TREINADO")
    print("="*70)
    
    # Configuração de caminhos
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    try:
        # Inicializa classificador
        classifier = InceptionALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Carrega modelo
        classifier.load_model(model_path)
        
        # Prepara dataset
        datasets = classifier.prepare_dataset(
            max_patches_per_patient=max_patches_per_patient,
            test_size=0.2,
            val_size=0.1
        )
        
        # Avalia modelo
        results = classifier.evaluate_model(
            test_df=datasets['test'],
            batch_size=16
        )
        
        # Visualizações
        classifier.plot_confusion_matrix(results['confusion_matrix'])
        
        print(f"✅ Avaliação concluída!")
        print(f"🎯 Acurácia: {results['accuracy']:.4f}")
        
        return classifier, results
        
    except Exception as e:
        print(f"❌ Erro durante avaliação: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Função principal com menu de opções"""
    
    print("Escolha o modo de execução:")
    print("1. Treinamento rápido (5 patches/paciente, 10 épocas)")
    print("2. Treinamento médio (10 patches/paciente, 30 épocas)")
    print("3. Treinamento completo (todos patches, 50 épocas)")
    print("4. Avaliar modelo existente")
    
    choice = input("\nDigite sua escolha (1-4): ").strip()
    
    if choice == "1":
        # Treinamento rápido
        classifier, history, results = run_inception_training(
            max_patches_per_patient=5,
            epochs=10,
            batch_size=8
        )
        
    elif choice == "2":
        # Treinamento médio
        classifier, history, results = run_inception_training(
            max_patches_per_patient=10,
            epochs=30,
            batch_size=8
        )
        
    elif choice == "3":
        # Treinamento completo
        confirm = input("⚠️ Isso pode demorar muitas horas. Continuar? (y/N): ")
        if confirm.lower() == 'y':
            classifier, history, results = run_inception_training(
                max_patches_per_patient=None,
                epochs=50,
                batch_size=4  # Batch menor para evitar problemas de memória
            )
        else:
            print("❌ Treinamento cancelado")
            return
            
    elif choice == "4":
        # Avaliação de modelo existente
        model_path = input("Digite o caminho do modelo (.h5): ").strip()
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}")
            return
            
        classifier, results = run_inception_evaluation(model_path)
        
    else:
        print("❌ Opção inválida")
        return
    
    print(f"\n🎉 Execução concluída!")

if __name__ == "__main__":
    main()
