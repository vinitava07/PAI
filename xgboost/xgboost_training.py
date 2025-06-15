# XGBoost Training Script
# Script para treinar modelo XGBoost com dados integrados

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.xgboost_integration import BreastCancerDataIntegration
import warnings
warnings.filterwarnings('ignore')

class XGBoostBreastCancerClassifier:
    """
    Classificador XGBoost para predição de metástase de linfonodos axilares
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_weights = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepara dados para treinamento
        
        Args:
            X: Features
            y: Target
            test_size: Proporção do conjunto de teste
            random_state: Semente aleatória
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Preparando dados para treinamento...")
        
        # Verifica distribuição das classes
        print(f"Distribuição original das classes: {np.bincount(y)}")
        
        # Split stratificado para manter proporção das classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Normalização das features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Conjunto de treino: {X_train_scaled.shape}")
        print(f"Conjunto de teste: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def calculate_class_weights(self, y):
        """
        Calcula pesos das classes para balanceamento
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        
        print(f"Pesos das classes: {self.class_weights}")
        return self.class_weights
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Treina modelo XGBoost
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        print("Treinando modelo XGBoost...")
        
        # Calcula pesos das classes
        self.calculate_class_weights(y_train)
        
        # Configuração do modelo XGBoost
        # Parâmetros conservadores para dados médicos
        xgb_params = {
            'objective': 'multi:softprob',  # Classificação multiclasse
            'num_class': 3,                 # 3 classes: N0, N+(1-2), N+(>2)
            'eval_metric': 'mlogloss',
            'max_depth': 4,                 # Evita overfitting
            'learning_rate': 0.05,          # Taxa de aprendizado conservadora
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,               # Regularização L1
            'reg_lambda': 1.0,              # Regularização L2
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Inicializa modelo
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # Cria sample_weight para balanceamento
        sample_weights = np.array([self.class_weights[y] for y in y_train])
        
        # Treinamento
        if X_val is not None and y_val is not None:
            # Com conjunto de validação
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'val'],
                verbose=False
            )
        else:
            # Sem conjunto de validação
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
            
        print("Treinamento concluído!")
        
    def evaluate_model(self, X_test, y_test, class_names=None):
        """
        Avalia modelo no conjunto de teste
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            class_names: Nomes das classes
            
        Returns:
            dict: Métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
            
        print("Avaliando modelo...")
        
        # Predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        # Nomes das classes
        if class_names is None:
            class_names = ['N0', 'N+(1-2)', 'N+(>2)']
            
        # Relatório de classificação
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Exibe resultados
        print(f"\nACURÁCIA: {accuracy:.4f}")
        print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return results
    
    def plot_feature_importance(self, feature_names=None, top_k=20):
        """
        Plota importância das features
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
            
        # Obtém importâncias
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        # Cria DataFrame para ordenação
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Seleciona top K features
        top_features = importance_df.head(top_k)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_k} Features Mais Importantes - XGBoost')
        plt.xlabel('Importância')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def plot_confusion_matrix(self, conf_matrix, class_names=None):
        """
        Plota matriz de confusão
        """
        if class_names is None:
            class_names = ['N0', 'N+(1-2)', 'N+(>2)']
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Matriz de Confusão - XGBoost')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Validação cruzada
        
        Args:
            X: Features
            y: Target
            cv_folds: Número de folds
            
        Returns:
            dict: Resultados da validação cruzada
        """
        if self.model is None:
            raise ValueError("Modelo não foi configurado ainda")
            
        print(f"Executando validação cruzada com {cv_folds} folds...")
        
        # StratifiedKFold para manter proporção das classes
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Escores
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=skf, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        print(f"Acurácia CV: {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"Scores individuais: {cv_scores}")
        
        return results
    
    def generate_scatter_plots(self, X, y, feature_names):
        """
        Gera gráficos de dispersão conforme especificado no projeto
        """
        print("Gerando gráficos de dispersão...")
        
        # Seleciona principais características morfológicas
        morpho_features = [
            'area_mean', 'circularity_mean', 
            'eccentricity_mean', 'normalized_nn_distance_mean'
        ]
        
        # Filtra features disponíveis
        available_morpho = [f for f in morpho_features if f in feature_names]
        
        if len(available_morpho) < 2:
            print("Poucas features morfológicas disponíveis para scatter plots")
            return
            
        # Cores para cada classe
        colors = ['black', 'blue', 'red']  # N0, N(1-2), N(>2)
        class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        
        # Cria DataFrame para facilitar plotting
        plot_data = pd.DataFrame(X, columns=feature_names)
        plot_data['class'] = y
        
        # Gera plots combinatórios
        n_features = len(available_morpho)
        fig, axes = plt.subplots(n_features-1, n_features-1, figsize=(15, 15))
        
        if n_features == 2:
            axes = [[axes]]
        elif n_features == 3:
            axes = [axes]
            
        for i in range(n_features-1):
            for j in range(i+1, n_features):
                ax = axes[i][j-1] if n_features > 2 else axes[i][j-1]
                
                feature_x = available_morpho[i]
                feature_y = available_morpho[j]
                
                # Plot por classe
                for class_idx in range(3):
                    class_data = plot_data[plot_data['class'] == class_idx]
                    
                    if len(class_data) > 0:
                        ax.scatter(
                            class_data[feature_x], 
                            class_data[feature_y],
                            c=colors[class_idx], 
                            label=class_names[class_idx],
                            alpha=0.6,
                            s=30
                        )
                
                ax.set_xlabel(feature_x.replace('_', ' ').title())
                ax.set_ylabel(feature_y.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Gráficos de Dispersão - Características Morfológicas', fontsize=16)
        plt.tight_layout()
        plt.show()


def main_training_pipeline():
    """
    Pipeline principal de treinamento
    """
    print("="*60)
    print("PIPELINE DE TREINAMENTO XGBOOST")
    print("="*60)
    
    # 1. Carrega dados integrados
    csv_path = "/home/leonardo/Documents/PUC/6. Semestre VI/Processamento e Análise de Imagens/PAI/patient-clinical-data.csv"
    patches_path = "/home/leonardo/Documents/PUC/6. Semestre VI/Processamento e Análise de Imagens/PAI/paper_patches/patches"
    
    # Integra dados (versão simplificada para exemplo)
    integrator = BreastCancerDataIntegration(csv_path, patches_path)
    
    # Processa alguns pacientes para demonstração
    integrator.process_all_patients(max_patients=20, max_patches_per_patient=2)
    integrator.integrate_data()
    X, y, feature_names = integrator.prepare_xgboost_data()
    
    # 2. Inicializa classificador
    classifier = XGBoostBreastCancerClassifier()
    
    # 3. Prepara dados
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    
    # 4. Treina modelo
    classifier.train_model(X_train, y_train)
    
    # 5. Avalia modelo
    results = classifier.evaluate_model(X_test, y_test)
    
    # 6. Visualizações
    importance_df = classifier.plot_feature_importance(feature_names)
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    classifier.generate_scatter_plots(X, y, feature_names)
    
    # 7. Validação cruzada
    cv_results = classifier.cross_validate(X, y)
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Acurácia final: {results['accuracy']:.4f}")
    print(f"Acurácia CV: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    print("="*60)
    
    return classifier, results, importance_df


if __name__ == "__main__":
    classifier, results, importance_df = main_training_pipeline()
