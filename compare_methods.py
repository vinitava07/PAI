import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from inception_pipeline import InceptionALNClassifier
from xgboost_training import XGBoostBreastCancerClassifier

class ALNMethodComparison:
    """
    Compara diferentes métodos de classificação ALN:
    - XGBoost com features morfológicos
    - Inception V4 com patches
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        self.patches_dir = patches_dir
        self.clinical_data_path = clinical_data_path
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        self.results = {}
        
    def prepare_patient_level_evaluation(self, inception_results, xgboost_results):
        """
        Agrega predições de patches para nível de paciente para comparação justa
        
        Args:
            inception_results: Resultados do Inception V4 (patch-level)
            xgboost_results: Resultados do XGBoost (patient-level)
            
        Returns:
            Dicionário com predições agregadas
        """
        
        # Para Inception: agrega predições por paciente usando voto majoritário
        inception_classifier = InceptionALNClassifier(self.patches_dir, self.clinical_data_path)
        
        # Prepara dataset para obter mapeamento patch-paciente
        datasets = inception_classifier.prepare_dataset(max_patches_per_patient=10)
        test_df = datasets['test']
        
        # Mapeia patches para pacientes
        patient_predictions = {}
        patient_true_labels = {}
        
        for idx, (_, row) in enumerate(test_df.iterrows()):
            patient_id = row['patient_id']
            true_label = row['class']
            pred_label = inception_results['predictions'][idx]
            
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []
                patient_true_labels[patient_id] = true_label
                
            patient_predictions[patient_id].append(pred_label)
        
        # Voto majoritário por paciente
        final_predictions = []
        final_true_labels = []
        patient_ids = []
        
        for patient_id in patient_predictions:
            # Voto majoritário
            patches_preds = patient_predictions[patient_id]
            final_pred = max(set(patches_preds), key=patches_preds.count)
            
            final_predictions.append(final_pred)
            final_true_labels.append(patient_true_labels[patient_id])
            patient_ids.append(patient_id)
        
        aggregated_inception = {
            'patient_ids': patient_ids,
            'predictions': np.array(final_predictions),
            'true_labels': np.array(final_true_labels),
            'method': 'Inception V4 (Agregado)'
        }
        
        # XGBoost já está em nível de paciente
        aggregated_xgboost = {
            'predictions': xgboost_results['predictions'],
            'true_labels': xgboost_results['true_labels'],
            'method': 'XGBoost'
        }
        
        return aggregated_inception, aggregated_xgboost
        
    def calculate_metrics(self, true_labels, predictions, method_name):
        """Calcula métricas detalhadas para um método"""
        
        accuracy = accuracy_score(true_labels, predictions)
        
        report = classification_report(
            true_labels, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
    def compare_methods(self, inception_model_path=None, xgboost_model_path=None):
        """
        Executa comparação completa entre métodos
        
        Args:
            inception_model_path: Caminho do modelo Inception treinado
            xgboost_model_path: Caminho do modelo XGBoost (opcional)
        """
        
        print("="*70)
        print("⚖️ COMPARAÇÃO DE MÉTODOS - ALN CLASSIFICATION")
        print("="*70)
        
        results = {}
        
        # 1. Avalia Inception V4
        if inception_model_path and Path(inception_model_path).exists():
            print("\n🧠 Avaliando Inception V4...")
            
            inception_classifier = InceptionALNClassifier(
                self.patches_dir, self.clinical_data_path
            )
            inception_classifier.load_model(inception_model_path)
            
            datasets = inception_classifier.prepare_dataset(max_patches_per_patient=10)
            inception_results = inception_classifier.evaluate_model(datasets['test'])
            
            results['inception_patch'] = self.calculate_metrics(
                inception_results['true_labels'],
                inception_results['predictions'],
                'Inception V4 (Patch-level)'
            )
            
        # 2. Avalia XGBoost
        print("\n🌳 Avaliando XGBoost...")
        try:
            from xgboost_integration import BreastCancerDataIntegration
            
            # Prepara dados para XGBoost
            integrator = BreastCancerDataIntegration(
                self.clinical_data_path, self.patches_dir
            )
            
            # Processa alguns pacientes para comparação
            integrator.process_all_patients(max_patients=100, max_patches_per_patient=5)
            integrator.integrate_data()
            X, y, features = integrator.prepare_xgboost_data()
            
            # Treina XGBoost
            xgb_classifier = XGBoostBreastCancerClassifier()
            X_train, X_test, y_train, y_test = xgb_classifier.prepare_data(X, y)
            xgb_classifier.train_model(X_train, y_train)
            xgb_results = xgb_classifier.evaluate_model(X_test, y_test)
            
            results['xgboost'] = self.calculate_metrics(
                y_test, xgb_results['predictions'], 'XGBoost'
            )
            
        except Exception as e:
            print(f"⚠️ Erro ao avaliar XGBoost: {str(e)}")
            
        # 3. Agrega resultados para comparação justa (nível de paciente)
        if 'inception_patch' in results and 'xgboost' in results:
            print("\n📊 Agregando para nível de paciente...")
            
            # Aqui você pode implementar agregação mais sofisticada
            # Por simplificidade, usa os resultados existentes
            
        # 4. Gera relatório comparativo
        self.generate_comparison_report(results)
        
        return results
        
    def generate_comparison_report(self, results):
        """Gera relatório comparativo detalhado"""
        
        print("\n" + "="*70)
        print("📊 RELATÓRIO COMPARATIVO")
        print("="*70)
        
        # Tabela de acurácias
        print("\n🎯 Acurácias por método:")
        print("-" * 40)
        for method_key, result in results.items():
            method_name = result['method']
            accuracy = result['accuracy']
            print(f"{method_name:25}: {accuracy:.4f}")
            
        # Relatório detalhado por classe
        print("\n📈 Métricas detalhadas por classe:")
        print("-" * 60)
        
        for method_key, result in results.items():
            print(f"\n{result['method']}:")
            report = result['classification_report']
            
            for class_name in self.class_names:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    support = report[class_name]['support']
                    print(f"  {class_name:8}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} N={support:3d}")
                    
        # Visualizações comparativas
        self.plot_comparison_charts(results)
        
    def plot_comparison_charts(self, results):
        """Gera gráficos comparativos"""
        
        n_methods = len(results)
        if n_methods == 0:
            return
            
        # 1. Comparação de acurácias
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(15, 5))
        
        # Gráfico de barras com acurácias
        methods = []
        accuracies = []
        
        for result in results.values():
            methods.append(result['method'])
            accuracies.append(result['accuracy'])
            
        axes[0].bar(methods, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(methods)])
        axes[0].set_title('Comparação de Acurácias')
        axes[0].set_ylabel('Acurácia')
        axes[0].set_ylim(0, 1)
        
        # Rotaciona labels se necessário
        if len(max(methods, key=len)) > 10:
            axes[0].tick_params(axis='x', rotation=45)
            
        # Adiciona valores nas barras
        for i, acc in enumerate(accuracies):
            axes[0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            
        # 2. Matrizes de confusão
        for idx, (method_key, result) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax
            )
            ax.set_title(f'Matriz de Confusão\n{result["method"]}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')
            
        plt.tight_layout()
        plt.show()
        
        # 3. Gráfico de métricas por classe
        self.plot_class_metrics_comparison(results)
        
    def plot_class_metrics_comparison(self, results):
        """Plota comparação de métricas por classe"""
        
        metrics = ['precision', 'recall', 'f1-score']
        n_metrics = len(metrics)
        n_classes = len(self.class_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(15, 5))
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            # Prepara dados para o gráfico
            x = np.arange(n_classes)
            width = 0.35
            
            method_names = []
            for idx, (method_key, result) in enumerate(results.items()):
                method_name = result['method']
                method_names.append(method_name)
                
                values = []
                for class_name in self.class_names:
                    if class_name in result['classification_report']:
                        values.append(result['classification_report'][class_name][metric])
                    else:
                        values.append(0)
                        
                ax.bar(x + idx * width, values, width, label=method_name, alpha=0.8)
                
            ax.set_xlabel('Classes')
            ax.set_ylabel(metric.title())
            ax.set_title(f'Comparação de {metric.title()} por Classe')
            ax.set_xticks(x + width * (len(results) - 1) / 2)
            ax.set_xticklabels(self.class_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.show()
        
    def save_comparison_results(self, results, output_path="method_comparison_results.csv"):
        """Salva resultados da comparação em CSV"""
        
        comparison_data = []
        
        for method_key, result in results.items():
            method_name = result['method']
            
            # Adiciona métricas gerais
            row = {
                'method': method_name,
                'overall_accuracy': result['accuracy'],
                'macro_avg_precision': result['classification_report']['macro avg']['precision'],
                'macro_avg_recall': result['classification_report']['macro avg']['recall'],
                'macro_avg_f1': result['classification_report']['macro avg']['f1-score']
            }
            
            # Adiciona métricas por classe
            for class_name in self.class_names:
                if class_name in result['classification_report']:
                    class_report = result['classification_report'][class_name]
                    row[f'{class_name}_precision'] = class_report['precision']
                    row[f'{class_name}_recall'] = class_report['recall']
                    row[f'{class_name}_f1'] = class_report['f1-score']
                    row[f'{class_name}_support'] = class_report['support']
                    
            comparison_data.append(row)
            
        # Salva em CSV
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(output_path, index=False)
        
        print(f"📄 Resultados salvos em: {output_path}")
        
        return df_comparison

# Função principal para execução
def run_method_comparison():
    """Executa comparação completa entre métodos"""
    
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    comparator = ALNMethodComparison(
        patches_dir=str(patches_dir),
        clinical_data_path=str(clinical_data_path)
    )
    
    # Busca por modelos treinados
    inception_model = base_dir / "best_inception_aln_model.h5"
    if not inception_model.exists():
        inception_model = base_dir / "inception_v4_aln_final.h5"
        
    if inception_model.exists():
        print(f"📁 Modelo Inception encontrado: {inception_model}")
    else:
        print("⚠️ Nenhum modelo Inception encontrado. Execute train_inception.py primeiro.")
        inception_model = None
        
    # Executa comparação
    results = comparator.compare_methods(
        inception_model_path=str(inception_model) if inception_model else None
    )
    
    # Salva resultados
    if results:
        comparator.save_comparison_results(results)
        
    return comparator, results

if __name__ == "__main__":
    comparator, results = run_method_comparison()
