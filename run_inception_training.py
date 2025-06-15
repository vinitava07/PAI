#!/usr/bin/env python3
"""
Script para executar o treinamento do Inception V3 com diferentes configurações
Permite escolher entre treino rápido, médio ou completo
"""

import argparse
from pathlib import Path
import sys
import time
from train_inception_v3_improved import InceptionV3ALNClassifier


def run_training(mode='rapido', max_patches=None, epochs=None, batch_size=16):
    """
    Executa treinamento com configurações pré-definidas
    
    Args:
        mode: Modo de treinamento ('rapido', 'medio', 'completo', 'custom')
        max_patches: Número máximo de patches por paciente
        epochs: Número de épocas
        batch_size: Tamanho do batch
    """
    # Configurações por modo
    configs = {
        'rapido': {
            'max_patches': 5,
            'epochs': 10,
            'fine_tune_epochs': 5,
            'fine_tune_layers': 50,
            'description': 'Treinamento rápido para testes (5 patches/paciente, 15 épocas total)'
        },
        'medio': {
            'max_patches': 10,
            'epochs': 30,
            'fine_tune_epochs': 15,
            'fine_tune_layers': 100,
            'description': 'Treinamento médio padrão (10 patches/paciente, 45 épocas total)'
        },
        'completo': {
            'max_patches': None,
            'epochs': 50,
            'fine_tune_epochs': 20,
            'fine_tune_layers': 150,
            'description': 'Treinamento completo (todos patches, 70 épocas total)'
        },
        'custom': {
            'max_patches': max_patches,
            'epochs': epochs or 30,
            'fine_tune_epochs': 15,
            'fine_tune_layers': 100,
            'description': f'Treinamento customizado ({max_patches or "todos"} patches/paciente, {epochs or 30} épocas)'
        }
    }
    
    if mode not in configs:
        print(f"❌ Modo inválido: {mode}")
        return
    
    config = configs[mode]
    
    print("="*80)
    print(f"🚀 INICIANDO TREINAMENTO - MODO: {mode.upper()}")
    print("="*80)
    print(f"\n📋 {config['description']}")
    print(f"  - Patches por paciente: {config['max_patches'] or 'Todos'}")
    print(f"  - Épocas (fase 1): {config['epochs']}")
    print(f"  - Épocas (fine-tuning): {config['fine_tune_epochs']}")
    print(f"  - Camadas para fine-tuning: {config['fine_tune_layers']}")
    print(f"  - Batch size: {batch_size}")
    
    # Confirmação para modo completo
    if mode == 'completo':
        print("\n⚠️  ATENÇÃO: O treinamento completo pode levar várias horas!")
        response = input("Deseja continuar? (s/N): ")
        if response.lower() != 's':
            print("❌ Treinamento cancelado")
            return
    
    # Inicia cronômetro
    start_time = time.time()
    
    # Caminhos
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    try:
        # Inicializa classificador
        classifier = InceptionV3ALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Carrega dados clínicos
        classifier.load_clinical_data()
        
        # Prepara dataset
        print("\n📊 Preparando dataset...")
        datasets = classifier.prepare_dataset(
            max_patches_per_patient=config['max_patches'],
            test_size=0.2,
            val_size=0.1
        )
        
        # Constrói modelo
        print("\n🏗️ Construindo modelo...")
        model = classifier.build_model(
            learning_rate=0.001,
            dropout_rate=0.5
        )
        
        # Treina modelo
        print("\n🎯 Iniciando treinamento...")
        history = classifier.train_model(
            datasets=datasets,
            epochs=config['epochs'],
            batch_size=batch_size,
            fine_tune=True,
            fine_tune_epochs=config['fine_tune_epochs'],
            fine_tune_layers=config['fine_tune_layers']
        )
        
        # Avalia no conjunto de teste
        print("\n📈 Avaliando modelo...")
        test_results = classifier.evaluate_model(
            test_df=datasets['test'],
            batch_size=batch_size
        )
        
        # Avalia por paciente
        print("\n👥 Avaliando por paciente...")
        patient_results = classifier.evaluate_by_patient(
            test_df=datasets['test'],
            aggregation='mean'
        )
        
        # Gera relatório final
        classifier.generate_final_report(
            history=history,
            test_results=test_results,
            patient_results=patient_results
        )
        
        # Tempo total
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "="*80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"\n⏱️ Tempo total: {hours}h {minutes}m {seconds}s")
        print(f"📁 Resultados salvos em: {classifier.run_dir}")
        print("\n📊 Resumo dos Resultados:")
        print(f"  - Acurácia no teste (patches): {test_results['accuracy']:.4f}")
        print(f"  - Acurácia por paciente: {patient_results['accuracy']:.4f}")
        print(f"  - AUC médio: {test_results['test_auc']:.4f}")
        
        # Resultados por classe
        print("\n📊 Resultados por classe:")
        for i, class_name in enumerate(classifier.class_names):
            print(f"\n  {class_name}:")
            print(f"    - Precisão: {test_results['precision'][i]:.3f}")
            print(f"    - Recall: {test_results['recall'][i]:.3f}")
            print(f"    - F1-Score: {test_results['f1_score'][i]:.3f}")
        
        return classifier, history, test_results, patient_results
        
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    """Função principal com argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Treina modelo Inception V3 para classificação ALN'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='medio',
        choices=['rapido', 'medio', 'completo', 'custom'],
        help='Modo de treinamento (default: medio)'
    )
    
    parser.add_argument(
        '--patches',
        type=int,
        default=None,
        help='Número máximo de patches por paciente (só para modo custom)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Número de épocas (só para modo custom)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamanho do batch (default: 16)'
    )
    
    args = parser.parse_args()
    
    # Validações
    if args.mode == 'custom' and args.epochs is None:
        print("⚠️ Modo custom requer --epochs")
        args.epochs = 30
        print(f"  Usando valor padrão: {args.epochs} épocas")
    
    # Executa treinamento
    run_training(
        mode=args.mode,
        max_patches=args.patches,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Menu interativo se executado sem argumentos
    if len(sys.argv) == 1:
        print("\n🧠 TREINAMENTO INCEPTION V3 - CLASSIFICAÇÃO ALN")
        print("="*60)
        print("\nEscolha o modo de treinamento:")
        print("1. Treinamento rápido (teste)")
        print("2. Treinamento médio (recomendado)")
        print("3. Treinamento completo (várias horas)")
        print("4. Sair")
        
        choice = input("\nOpção (1-4): ").strip()
        
        if choice == '1':
            run_training(mode='rapido')
        elif choice == '2':
            run_training(mode='medio')
        elif choice == '3':
            run_training(mode='completo')
        elif choice == '4':
            print("👋 Saindo...")
        else:
            print("❌ Opção inválida")
    else:
        # Executa com argumentos de linha de comando
        main()
