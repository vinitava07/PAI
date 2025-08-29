import os
import sys
from pathlib import Path
import tensorflow as tf

# Configura GPU se disponível (mesma configuração do Inception)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_limit(physical_devices[0], 5500)
    print(f"GPU disponível: {physical_devices[0]}")
else:
    print("Executando em CPU")

# Importa o pipeline do MobileNet
from mobilenet_pipeline import MobileNetALNClassifier

# Define caminhos (mesma estrutura do Inception)
base_dir = Path(__file__).parent.parent
patches_dir = base_dir / "patches"
clinical_data_path = base_dir / "patient-clinical-data.csv"

def run_mobilenet_training(max_patches_per_patient=10, epochs=30, fine_tune_epochs=10, batch_size=32):
    """
    Executa treinamento completo do MobileNetV2 pré-treinado
    Nota: MobileNet é mais eficiente, então podemos usar batch_size maior
    
    Args:
        max_patches_per_patient: Máximo de patches por paciente
        epochs: Número de épocas de feature extraction
        fine_tune_epochs: Número de épocas de fine-tuning
        batch_size: Tamanho do batch (maior que Inception pois MobileNet é mais leve)
    """
    
    print("="*80)
    print("📱 TREINAMENTO MOBILENET V2 PRÉ-TREINADO - CLASSIFICAÇÃO ALN")
    print("="*80)
        
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
    print(f"🔄 Épocas feature extraction: {epochs}")
    print(f"🎯 Épocas fine-tuning: {fine_tune_epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📐 Input size: 224x224 (MobileNet padrão)")
    
    try:
        # Inicializa classificador MobileNet
        classifier = MobileNetALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Prepara dataset
        print("\n" + "="*60)
        print("📊 PREPARAÇÃO DOS DADOS")
        print("="*60)
        
        datasets = classifier.prepare_dataset(
            max_patches_per_patient=max_patches_per_patient,
            test_size=0.2  # 80/20 split como no Inception
        )
        
        # Constrói modelo
        print("\n" + "="*60)
        print("🏗️ CONSTRUÇÃO DO MODELO MOBILENET V2")
        print("="*60)
        
        classifier.build_model(
            fine_tune=True,
            freeze_layers=80  # MobileNet tem menos camadas (~155), então congelamos menos
        )
        
        # Treina modelo
        print("\n" + "="*60)
        print("🚀 TREINAMENTO (TWO-STAGE)")
        print("="*60)
        
        history = classifier.train_model(
            datasets=datasets,
            epochs=epochs,
            batch_size=batch_size,
            fine_tune_epochs=fine_tune_epochs
        )
        
        # Avalia modelo
        print("\n" + "="*60)
        print("📈 AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("="*60)
        
        results = classifier.evaluate_model(
            test_df=datasets['test'],
            batch_size=batch_size
        )
        
        # Visualizações
        print("\n" + "="*60)
        print("📊 VISUALIZAÇÕES")
        print("="*60)
        
        classifier.plot_training_history(history)
        classifier.plot_confusion_matrix(results['confusion_matrix'])
        
        # Salva modelo
        model_path = base_dir / "mobilenet_v2_aln_final.h5"
        classifier.save_model(str(model_path))
        
        # Relatório final
        print("\n" + "="*60)
        print("📋 RELATÓRIO FINAL")
        print("="*60)
        
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
        
        # Informações do modelo
        print(f"\n🏗️ Informações do modelo:")
        total_params = classifier.model.count_params()
        print(f"  Total de parâmetros: {total_params:,}")
        print(f"  Modelo base: MobileNetV2 pré-treinado (ImageNet)")
        print(f"  Input size: 224x224 (vs 299x299 do Inception)")
        print(f"  Estratégia: Two-stage training (feature extraction + fine-tuning)")
        print(f"  Vantagens: Mais rápido, menor uso de memória, ideal para mobile")
        
        return classifier, history, results
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_mobilenet_evaluation(model_path, max_patches_per_patient=None):
    """
    Executa apenas avaliação com modelo MobileNet pré-treinado
    
    Args:
        model_path: Caminho do modelo salvo
        max_patches_per_patient: Máximo de patches por paciente
    """
    
    print("="*80)
    print("🔍 AVALIAÇÃO MOBILENET V2 - MODELO PRÉ-TREINADO")
    print("="*80)
    
    try:
        # Inicializa classificador
        classifier = MobileNetALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Carrega modelo
        classifier.load_model(model_path)
        
        # Prepara dataset
        datasets = classifier.prepare_dataset(
            max_patches_per_patient=max_patches_per_patient,
            test_size=0.2
        )
        
        # Avalia modelo
        results = classifier.evaluate_model(
            test_df=datasets['test'],
            batch_size=32  # Batch maior para MobileNet
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
    
    print("\n" + "="*60)
    print("📱 MOBILENET V2 - CLASSIFICAÇÃO DE METÁSTASE ALN")
    print("="*60)
    print("\nEscolha o modo de execução:")
    print("1. Treinamento rápido (5 patches/paciente, 20 épocas)")
    print("2. Treinamento médio (10 patches/paciente, 30 épocas)")
    print("3. Treinamento completo (todos patches, 50 épocas)")
    print("4. Avaliar modelo existente")
    print("5. Comparação rápida MobileNet vs Inception")
    
    choice = input("\n👉 Digite sua escolha (1-5): ").strip()
    
    if choice == "1":
        # Treinamento rápido - ideal para testes
        print("\n⚡ Modo: Treinamento Rápido")
        classifier, history, results = run_mobilenet_training(
            max_patches_per_patient=5,
            epochs=20,
            fine_tune_epochs=5,
            batch_size=32  # Batch maior que Inception (16)
        )
        
    elif choice == "2":
        # Treinamento médio - bom equilíbrio
        print("\n⚖️ Modo: Treinamento Médio")
        classifier, history, results = run_mobilenet_training(
            max_patches_per_patient=10,
            epochs=30,
            fine_tune_epochs=10,
            batch_size=24
        )
        
    elif choice == "3":
        # Treinamento completo
        print("\n🔥 Modo: Treinamento Completo")
        confirm = input("⚠️ Isso pode demorar várias horas. Continuar? (y/N): ")
        if confirm.lower() == 'y':
            classifier, history, results = run_mobilenet_training(
                max_patches_per_patient=None,  # Todos os patches
                epochs=40,
                fine_tune_epochs=15,
                batch_size=16  # Batch menor para evitar problemas de memória
            )
        else:
            print("❌ Treinamento cancelado")
            return
            
    elif choice == "4":
        # Avaliação de modelo existente
        print("\n🔍 Modo: Avaliação de Modelo")
        model_path = input("Digite o caminho do modelo (.h5): ").strip()
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}")
            return
            
        classifier, results = run_mobilenet_evaluation(model_path)
        
    elif choice == "5":
        # Comparação com Inception
        print("\n📊 Comparação MobileNet vs Inception:")
        print("\n🏗️ Arquitetura:")
        print("  MobileNet V2: ~3.5M parâmetros, 224x224 input")
        print("  Inception V3: ~23.8M parâmetros, 299x299 input")
        print("\n⚡ Performance:")
        print("  MobileNet: ~7x mais rápido no treinamento")
        print("  MobileNet: ~4x menos uso de memória")
        print("\n🎯 Acurácia esperada:")
        print("  MobileNet: 85-90% (depende do dataset)")
        print("  Inception: 87-93% (margem pequena)")
        print("\n💡 Recomendação:")
        print("  Use MobileNet para prototipagem rápida e deploy mobile")
        print("  Use Inception para máxima acurácia em servidores")
        return
        
    else:
        print("❌ Opção inválida")
        return
    
    print(f"\n🎉 Execução concluída!")

if __name__ == "__main__":
    main()
