# Pipeline Principal - Execução Simplificada
# Execute este arquivo para processar dados e treinar XGBoost

import os
import sys
from pathlib import Path

# Adiciona diretório atual ao path para imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from xgboost_integration import BreastCancerDataIntegration
from xgboost_training import XGBoostBreastCancerClassifier
import pandas as pd
import numpy as np

def run_complete_pipeline(max_patients=None, max_patches=3, quick_test=False):
    """
    Executa pipeline completo de processamento e treinamento
    
    Args:
        max_patients: Número máximo de pacientes (None = todos)
        max_patches: Máximo de patches por paciente  
        quick_test: Se True, executa apenas teste rápido
    """
    
    print("="*70)
    print("🔬 PIPELINE COMPLETO: CÂNCER DE MAMA + XGBOOST")
    print("="*70)
    
    # Configuração de paths
    base_dir = Path(__file__).parent
    csv_path = base_dir / "patient-clinical-data.csv"
    patches_path = base_dir / "paper_patches" / "patches"
    
    # Verifica se arquivos existem
    if not csv_path.exists():
        print(f"❌ Arquivo CSV não encontrado: {csv_path}")
        return None
        
    if not patches_path.exists():
        print(f"❌ Diretório de patches não encontrado: {patches_path}")
        return None
    
    # Configuração para teste rápido
    if quick_test:
        max_patients = 10
        max_patches = 2
        print("🚀 Modo teste rápido ativado!")
        
    print(f"📊 Processando até {max_patients or 'todos'} pacientes")
    print(f"🖼️ Máximo {max_patches} patches por paciente")
    
    try:
        # ETAPA 1: Integração de Dados
        print("\n" + "="*50)
        print("📋 ETAPA 1: INTEGRAÇÃO DE DADOS")
        print("="*50)
        
        integrator = BreastCancerDataIntegration(str(csv_path), str(patches_path))
        
        # Processa imagens
        integrator.process_all_patients(
            max_patients=max_patients,
            max_patches_per_patient=max_patches
        )
        
        # Integra dados
        dataset = integrator.integrate_data()
        X, y, feature_names = integrator.prepare_xgboost_data()
        
        # Exporta dataset
        output_file = base_dir / "integrated_dataset.csv"
        integrator.export_dataset(str(output_file))
        
        print(f"✅ Dataset integrado salvo: {output_file}")
        print(f"📈 Total de amostras: {len(X)}")
        print(f"🔢 Total de features: {len(feature_names)}")
        
        # ETAPA 2: Treinamento XGBoost
        print("\n" + "="*50)
        print("🤖 ETAPA 2: TREINAMENTO XGBOOST")
        print("="*50)
        
        # Verifica se há dados suficientes
        if len(X) < 10:
            print("⚠️ Poucos dados para treinamento. Aumente max_patients.")
            return integrator, None, None
            
        classifier = XGBoostBreastCancerClassifier()
        
        # Prepara dados
        X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
        
        # Treina modelo
        classifier.train_model(X_train, y_train)
        
        # Avalia modelo
        results = classifier.evaluate_model(X_test, y_test)
        
        # ETAPA 3: Visualizações
        print("\n" + "="*50)
        print("📊 ETAPA 3: VISUALIZAÇÕES")
        print("="*50)
        
        # Importância das features
        importance_df = classifier.plot_feature_importance(feature_names, top_k=15)
        
        # Matriz de confusão
        classifier.plot_confusion_matrix(results['confusion_matrix'])
        
        # Scatter plots (conforme projeto)
        classifier.generate_scatter_plots(X, y, feature_names)
        
        # Validação cruzada
        cv_results = classifier.cross_validate(X, y, cv_folds=5)
        
        # ETAPA 4: Relatório Final
        print("\n" + "="*50)
        print("📋 RELATÓRIO FINAL")
        print("="*50)
        
        print(f"🎯 Acurácia no teste: {results['accuracy']:.4f}")
        print(f"🔄 Acurácia CV: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
        
        # Top 5 features mais importantes
        print(f"\n🏆 Top 5 Features Mais Importantes:")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
        # Distribuição das classes
        unique, counts = np.unique(y, return_counts=True)
        class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        print(f"\n📊 Distribuição das Classes:")
        for class_idx, count in zip(unique, counts):
            print(f"  {class_names[class_idx]}: {count} pacientes")
            
        print("\n✅ Pipeline executado com sucesso!")
        
        return integrator, classifier, results
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {str(e)}")
        print("💡 Dica: Tente executar em modo teste rápido primeiro")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """
    Função principal - escolha o modo de execução
    """
    print("Escolha o modo de execução:")
    print("1. Teste rápido (10 pacientes, 2 patches cada)")
    print("2. Execução limitada (50 pacientes, 3 patches cada)")
    print("3. Execução completa (todos os pacientes)")
    
    choice = input("\nDigite sua escolha (1-3): ").strip()
    
    if choice == "1":
        # Teste rápido
        integrator, classifier, results = run_complete_pipeline(quick_test=True)
        
    elif choice == "2":
        # Execução limitada
        integrator, classifier, results = run_complete_pipeline(
            max_patients=50, 
            max_patches=3
        )
        
    elif choice == "3":
        # Execução completa
        confirm = input("⚠️ Isso pode demorar várias horas. Continuar? (y/N): ")
        if confirm.lower() == 'y':
            integrator, classifier, results = run_complete_pipeline()
        else:
            print("❌ Execução cancelada")
            return
            
    else:
        print("❌ Opção inválida")
        return
    
    if integrator is not None:
        print(f"\n🎉 Execução concluída!")
        print(f"📁 Resultados salvos no diretório atual")


if __name__ == "__main__":
    main()
