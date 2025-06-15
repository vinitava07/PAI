#!/usr/bin/env python3
"""
Script aprimorado para treinamento do Inception V3 para classificação ALN
Implementa corretamente a divisão 80/20 por paciente e validação durante treino
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, CSVLogger, LearningRateScheduler
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuração da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU disponível: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"⚠️ Erro ao configurar GPU: {e}")
else:
    print("⚠️ Nenhuma GPU encontrada - executando em CPU")


class InceptionV3ALNClassifier:
    """
    Classificador aprimorado para ALN status usando Inception V3
    """
    
    def __init__(self, patches_dir: str, clinical_data_path: str, output_dir: str = "models"):
        """
        Inicializa o classificador
        
        Args:
            patches_dir: Diretório com patches organizados por paciente
            clinical_data_path: Caminho do CSV com dados clínicos
            output_dir: Diretório para salvar modelos e resultados
        """
        self.patches_dir = Path(patches_dir)
        self.clinical_data_path = Path(clinical_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurações do modelo
        self.input_size = (299, 299)  # Tamanho padrão do Inception V3
        self.num_classes = 3
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        self.aln_mapping = {'N0': 0, 'N+(1-2)': 1, 'N+(>2)': 2}
        
        # Modelo e dados
        self.model = None
        self.clinical_data = None
        self.class_weights = None
        
        # Timestamp para organização dos resultados
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Diretório de saída: {self.run_dir}")
        
    def load_clinical_data(self) -> pd.DataFrame:
        """
        Carrega e processa dados clínicos do CSV
        """
        print("\n📋 Carregando dados clínicos...")
        
        # Carrega CSV
        self.clinical_data = pd.read_csv(self.clinical_data_path)
        
        # Converte Patient ID para string
        self.clinical_data['Patient ID'] = self.clinical_data['Patient ID'].astype(str)
        
        # Mapeia ALN status para códigos numéricos
        self.clinical_data['ALN_encoded'] = self.clinical_data['ALN status'].map(self.aln_mapping)
        
        # Remove pacientes sem ALN status
        initial_count = len(self.clinical_data)
        self.clinical_data = self.clinical_data.dropna(subset=['ALN_encoded'])
        removed_count = initial_count - len(self.clinical_data)
        
        if removed_count > 0:
            print(f"⚠️ Removidos {removed_count} pacientes sem ALN status")
        
        # Estatísticas
        print(f"✅ Total de pacientes: {len(self.clinical_data)}")
        print("\n📊 Distribuição das classes:")
        for class_name, count in self.clinical_data['ALN status'].value_counts().items():
            percentage = (count / len(self.clinical_data)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return self.clinical_data
    
    def prepare_dataset(self, max_patches_per_patient: Optional[int] = None, 
                       test_size: float = 0.2, 
                       val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Prepara dataset com divisão correta por paciente
        
        Args:
            max_patches_per_patient: Limite de patches por paciente (None = todos)
            test_size: Proporção para teste (padrão 20%)
            val_size: Proporção para validação do treino
            
        Returns:
            Dicionário com DataFrames para treino, validação e teste
        """
        print(f"\n📁 Preparando dataset de patches...")
        print(f"  Max patches/paciente: {max_patches_per_patient or 'Todos'}")
        
        # Coleta informações dos patches
        patch_data = []
        patient_patch_counts = {}
        
        for _, patient_row in self.clinical_data.iterrows():
            patient_id = str(patient_row['Patient ID'])
            aln_class = int(patient_row['ALN_encoded'])
            patient_dir = self.patches_dir / patient_id
            
            if not patient_dir.exists():
                print(f"⚠️ Diretório não encontrado: {patient_dir}")
                continue
            
            # Lista patches do paciente
            patch_files = sorted(list(patient_dir.glob("*.jpg")))
            
            if not patch_files:
                print(f"⚠️ Nenhum patch encontrado para paciente {patient_id}")
                continue
            
            # Limita número de patches se especificado
            if max_patches_per_patient and len(patch_files) > max_patches_per_patient:
                # Seleciona aleatoriamente para evitar viés
                np.random.seed(42)  # Reprodutibilidade
                indices = np.random.choice(len(patch_files), max_patches_per_patient, replace=False)
                patch_files = [patch_files[i] for i in sorted(indices)]
            
            patient_patch_counts[patient_id] = len(patch_files)
            
            # Adiciona patches ao dataset
            for patch_file in patch_files:
                patch_data.append({
                    'patch_path': str(patch_file),
                    'patient_id': patient_id,
                    'class': aln_class,
                    'class_name': self.class_names[aln_class]
                })
        
        if not patch_data:
            raise ValueError("❌ Nenhum patch foi encontrado!")
        
        # Converte para DataFrame
        df_patches = pd.DataFrame(patch_data)
        
        print(f"\n📊 Estatísticas dos patches:")
        print(f"  Total de patches: {len(df_patches)}")
        print(f"  Pacientes com patches: {len(patient_patch_counts)}")
        print(f"  Média patches/paciente: {np.mean(list(patient_patch_counts.values())):.1f}")
        
        # Distribuição por classe
        print("\n📊 Distribuição de patches por classe:")
        for class_name, count in df_patches['class_name'].value_counts().items():
            percentage = (count / len(df_patches)) * 100
            print(f"  {class_name}: {count} patches ({percentage:.1f}%)")
        
        # DIVISÃO POR PACIENTE (não por patch!)
        print(f"\n🔄 Dividindo dataset por paciente...")
        
        # Lista única de pacientes e suas classes
        unique_patients = df_patches['patient_id'].unique()
        patient_classes = []
        
        for pid in unique_patients:
            patient_class = df_patches[df_patches['patient_id'] == pid]['class'].iloc[0]
            patient_classes.append(patient_class)
        
        # Primeira divisão: 80% treino+val, 20% teste
        patients_train_val, patients_test, y_train_val, y_test = train_test_split(
            unique_patients, 
            patient_classes,
            test_size=test_size,
            random_state=42,
            stratify=patient_classes
        )
        
        # Segunda divisão: separa validação do treino
        val_split = val_size / (1 - test_size)  # Ajusta proporção
        patients_train, patients_val, y_train, y_val = train_test_split(
            patients_train_val,
            y_train_val,
            test_size=val_split,
            random_state=42,
            stratify=y_train_val
        )
        
        # Cria DataFrames finais
        train_patches = df_patches[df_patches['patient_id'].isin(patients_train)]
        val_patches = df_patches[df_patches['patient_id'].isin(patients_val)]
        test_patches = df_patches[df_patches['patient_id'].isin(patients_test)]
        
        # Embaralha patches dentro de cada conjunto
        train_patches = train_patches.sample(frac=1, random_state=42).reset_index(drop=True)
        val_patches = val_patches.sample(frac=1, random_state=42).reset_index(drop=True)
        test_patches = test_patches.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Estatísticas finais
        print(f"\n✅ Divisão concluída:")
        print(f"  Pacientes treino: {len(patients_train)} ({len(train_patches)} patches)")
        print(f"  Pacientes validação: {len(patients_val)} ({len(val_patches)} patches)")
        print(f"  Pacientes teste: {len(patients_test)} ({len(test_patches)} patches)")
        
        # Verifica distribuição de classes
        print("\n📊 Distribuição de classes por conjunto:")
        for split_name, split_df in [('Treino', train_patches), 
                                     ('Validação', val_patches), 
                                     ('Teste', test_patches)]:
            dist = split_df['class_name'].value_counts()
            print(f"\n  {split_name}:")
            for class_name in self.class_names:
                count = dist.get(class_name, 0)
                percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # Salva informações da divisão
        split_info = {
            'timestamp': self.timestamp,
            'total_patches': len(df_patches),
            'total_patients': len(unique_patients),
            'max_patches_per_patient': max_patches_per_patient,
            'test_size': test_size,
            'val_size': val_size,
            'train_patients': len(patients_train),
            'val_patients': len(patients_val),
            'test_patients': len(patients_test),
            'train_patches': len(train_patches),
            'val_patches': len(val_patches),
            'test_patches': len(test_patches),
            'class_distribution': {
                'train': train_patches['class_name'].value_counts().to_dict(),
                'val': val_patches['class_name'].value_counts().to_dict(),
                'test': test_patches['class_name'].value_counts().to_dict()
            }
        }
        
        with open(self.run_dir / 'dataset_split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'all': df_patches
        }
    
    def create_data_generator(self, df_patches: pd.DataFrame, 
                            batch_size: int = 16,
                            shuffle: bool = True,
                            augment: bool = False) -> tf.keras.utils.Sequence:
        """
        Cria gerador de dados otimizado
        
        Args:
            df_patches: DataFrame com informações dos patches
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar
            augment: Se deve aplicar data augmentation
            
        Returns:
            Gerador de dados
        """
        # Configuração de data augmentation para imagens histológicas
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=90,           # Rotação aleatória
                width_shift_range=0.1,       # Deslocamento horizontal
                height_shift_range=0.1,      # Deslocamento vertical
                horizontal_flip=True,        # Flip horizontal
                vertical_flip=True,          # Flip vertical
                zoom_range=0.1,              # Zoom aleatório
                fill_mode='reflect',         # Modo de preenchimento
                preprocessing_function=preprocess_input  # Pré-processamento Inception V3
            )
        else:
            datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )
        
        # Cria gerador
        generator = datagen.flow_from_dataframe(
            dataframe=df_patches,
            x_col='patch_path',
            y_col='class_name',
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=shuffle,
            seed=42
        )
        
        return generator
    
    def build_model(self, learning_rate: float = 0.001,
                   dropout_rate: float = 0.5) -> Model:
        """
        Constrói modelo Inception V3 com camadas customizadas
        
        Args:
            learning_rate: Taxa de aprendizado inicial
            dropout_rate: Taxa de dropout
            
        Returns:
            Modelo compilado
        """
        print("\n🏗️ Construindo modelo Inception V3...")
        
        # Modelo base pré-treinado
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Congela modelo base inicialmente
        base_model.trainable = False
        
        # Adiciona camadas de classificação
        inputs = Input(shape=(299, 299, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate * 0.7)(x)  # Dropout menor na segunda camada
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Cria modelo
        self.model = Model(inputs, outputs)
        
        # Compila modelo
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Informações do modelo
        total_params = self.model.count_params()
        trainable_params = np.sum([tf.keras.backend.count_params(w) 
                                  for w in self.model.trainable_weights])
        
        print(f"✅ Modelo construído:")
        print(f"  Total de parâmetros: {total_params:,}")
        print(f"  Parâmetros treináveis: {trainable_params:,}")
        print(f"  Taxa de aprendizado: {learning_rate}")
        print(f"  Dropout: {dropout_rate}")
        
        return self.model
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calcula pesos balanceados para as classes
        """
        y_train = train_df['class'].values
        classes = np.unique(y_train)
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        self.class_weights = dict(zip(classes, weights))
        
        print("\n⚖️ Pesos das classes calculados:")
        for class_idx, weight in self.class_weights.items():
            class_name = self.class_names[class_idx]
            print(f"  {class_name}: {weight:.3f}")
        
        return self.class_weights
    
    def train_model(self, datasets: Dict[str, pd.DataFrame],
                   epochs: int = 50,
                   batch_size: int = 16,
                   fine_tune: bool = True,
                   fine_tune_epochs: int = 20,
                   fine_tune_layers: int = 100) -> tf.keras.callbacks.History:
        """
        Treina modelo com estratégia two-stage
        
        Args:
            datasets: Dicionário com DataFrames de treino, validação e teste
            epochs: Épocas para primeira fase
            batch_size: Tamanho do batch
            fine_tune: Se deve fazer fine-tuning
            fine_tune_epochs: Épocas para fine-tuning
            fine_tune_layers: Número de camadas a descongelar
            
        Returns:
            Histórico de treinamento
        """
        print("\n🚀 Iniciando treinamento...")
        
        # Prepara dados
        train_df = datasets['train']
        val_df = datasets['val']
        
        # Calcula pesos das classes
        class_weights = self.calculate_class_weights(train_df)
        
        # Cria geradores
        train_generator = self.create_data_generator(
            train_df, batch_size=batch_size, shuffle=True, augment=True
        )
        val_generator = self.create_data_generator(
            val_df, batch_size=batch_size, shuffle=False, augment=False
        )
        
        # Callbacks para primeira fase
        callbacks_phase1 = [
            ModelCheckpoint(
                str(self.run_dir / 'best_model_phase1.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(str(self.run_dir / 'training_phase1.csv')),
            TensorBoard(
                log_dir=str(self.run_dir / 'tensorboard_phase1'),
                histogram_freq=1
            )
        ]
        
        # FASE 1: Feature Extraction
        print("\n" + "="*60)
        print("📍 FASE 1: FEATURE EXTRACTION (modelo base congelado)")
        print("="*60)
        
        history1 = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        # Salva gráficos da fase 1
        self._plot_training_curves(history1, phase=1)
        
        # FASE 2: Fine-tuning (opcional)
        if fine_tune and fine_tune_epochs > 0:
            print("\n" + "="*60)
            print("📍 FASE 2: FINE-TUNING (descongelando camadas)")
            print("="*60)
            
            # Descongela últimas camadas do modelo base
            base_model = self.model.layers[1]  # InceptionV3 é a segunda camada
            base_model.trainable = True
            
            # Congela todas exceto as últimas 'fine_tune_layers'
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            
            print(f"✅ Descongeladas {fine_tune_layers} últimas camadas")
            
            # Recompila com learning rate menor
            self.model.compile(
                optimizer=Adam(learning_rate=0.00001),  # LR muito menor
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            # Callbacks para fine-tuning
            callbacks_phase2 = [
                ModelCheckpoint(
                    str(self.run_dir / 'best_model_final.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=3,
                    min_lr=1e-8,
                    verbose=1
                ),
                CSVLogger(str(self.run_dir / 'training_phase2.csv')),
                TensorBoard(
                    log_dir=str(self.run_dir / 'tensorboard_phase2'),
                    histogram_freq=1
                )
            ]
            
            # Treina com fine-tuning
            history2 = self.model.fit(
                train_generator,
                epochs=fine_tune_epochs,
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=callbacks_phase2,
                verbose=1
            )
            
            # Combina históricos
            for key in history1.history.keys():
                history1.history[key].extend(history2.history[key])
            
            # Salva gráficos combinados
            self._plot_training_curves(history1, phase='combined')
        
        print("\n✅ Treinamento concluído!")
        
        # Salva modelo final
        final_model_path = self.run_dir / 'inception_v3_aln_final.h5'
        self.model.save(str(final_model_path))
        print(f"💾 Modelo final salvo: {final_model_path}")
        
        return history1
    
    def evaluate_model(self, test_df: pd.DataFrame, 
                      batch_size: int = 16) -> Dict:
        """
        Avalia modelo no conjunto de teste
        
        Args:
            test_df: DataFrame com dados de teste
            batch_size: Tamanho do batch
            
        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n📊 Avaliando modelo no conjunto de teste...")
        
        # Cria gerador de teste (sem augmentation)
        test_generator = self.create_data_generator(
            test_df, batch_size=batch_size, shuffle=False, augment=False
        )
        
        # Avaliação com o modelo
        test_loss, test_acc, test_auc = self.model.evaluate(
            test_generator, verbose=1
        )
        
        # Predições detalhadas
        print("\n🔮 Gerando predições...")
        test_generator.reset()  # Reset para garantir ordem
        predictions = self.model.predict(test_generator, verbose=1)
        
        # Labels verdadeiros
        true_labels = test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Métricas detalhadas
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None
        )
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # ROC AUC para cada classe
        roc_auc_scores = {}
        for i in range(self.num_classes):
            y_true_binary = (true_labels == i).astype(int)
            y_score = predictions[:, i]
            roc_auc_scores[self.class_names[i]] = roc_auc_score(y_true_binary, y_score)
        
        # Resultados
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_auc_scores': roc_auc_scores,
            'predictions': predictions,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
        
        # Exibe resultados
        print(f"\n📈 Resultados no conjunto de teste:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Acurácia: {accuracy:.4f}")
        print(f"  AUC médio: {test_auc:.4f}")
        
        print("\n📊 Métricas por classe:")
        for i, class_name in enumerate(self.class_names):
            print(f"\n  {class_name}:")
            print(f"    Precisão: {precision[i]:.3f}")
            print(f"    Recall: {recall[i]:.3f}")
            print(f"    F1-Score: {f1[i]:.3f}")
            print(f"    Suporte: {support[i]}")
            print(f"    ROC AUC: {roc_auc_scores[class_name]:.3f}")
        
        # Salva resultados
        self._save_evaluation_results(results)
        
        # Gera visualizações
        self._plot_confusion_matrix(conf_matrix)
        self._plot_roc_curves(true_labels, predictions)
        
        return results
    
    def evaluate_by_patient(self, test_df: pd.DataFrame, 
                          aggregation: str = 'mean') -> Dict:
        """
        Avalia modelo agregando predições por paciente
        
        Args:
            test_df: DataFrame com dados de teste
            aggregation: Método de agregação ('mean', 'max', 'vote')
            
        Returns:
            Dicionário com métricas por paciente
        """
        print(f"\n👥 Avaliando por paciente (agregação: {aggregation})...")
        
        patient_results = []
        
        # Agrupa patches por paciente
        for patient_id, patient_patches in test_df.groupby('patient_id'):
            # Predições para todos os patches do paciente
            patch_predictions = []
            
            for _, patch in patient_patches.iterrows():
                img = load_img(patch['patch_path'], target_size=self.input_size)
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                
                pred = self.model.predict(img, verbose=0)
                patch_predictions.append(pred[0])
            
            patch_predictions = np.array(patch_predictions)
            
            # Agregação
            if aggregation == 'mean':
                patient_pred = np.mean(patch_predictions, axis=0)
            elif aggregation == 'max':
                patient_pred = np.max(patch_predictions, axis=0)
            elif aggregation == 'vote':
                patch_classes = np.argmax(patch_predictions, axis=1)
                patient_class = np.bincount(patch_classes).argmax()
                patient_pred = np.zeros(self.num_classes)
                patient_pred[patient_class] = 1.0
            else:
                raise ValueError(f"Método de agregação inválido: {aggregation}")
            
            # Classe verdadeira do paciente
            true_class = patient_patches.iloc[0]['class']
            
            patient_results.append({
                'patient_id': patient_id,
                'true_class': true_class,
                'predicted_class': np.argmax(patient_pred),
                'prediction_probs': patient_pred,
                'num_patches': len(patient_patches)
            })
        
        # Calcula métricas por paciente
        patient_df = pd.DataFrame(patient_results)
        
        patient_accuracy = accuracy_score(
            patient_df['true_class'], 
            patient_df['predicted_class']
        )
        
        patient_report = classification_report(
            patient_df['true_class'],
            patient_df['predicted_class'],
            target_names=self.class_names,
            output_dict=True
        )
        
        patient_conf_matrix = confusion_matrix(
            patient_df['true_class'],
            patient_df['predicted_class']
        )
        
        print(f"\n📊 Resultados por paciente:")
        print(f"  Total de pacientes: {len(patient_df)}")
        print(f"  Acurácia: {patient_accuracy:.4f}")
        
        # Salva matriz de confusão por paciente
        self._plot_confusion_matrix(
            patient_conf_matrix, 
            title=f'Matriz de Confusão - Por Paciente ({aggregation})',
            filename=f'confusion_matrix_patient_{aggregation}.png'
        )
        
        return {
            'patient_results': patient_df,
            'accuracy': patient_accuracy,
            'classification_report': patient_report,
            'confusion_matrix': patient_conf_matrix
        }
    
    def _plot_training_curves(self, history: tf.keras.callbacks.History, 
                            phase: str = '1'):
        """Plota curvas de treinamento"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Acurácia
        axes[0].plot(history.history['accuracy'], label='Treino', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
        axes[0].set_title(f'Acurácia - Fase {phase}', fontsize=14)
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Acurácia')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Treino', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validação', linewidth=2)
        axes[1].set_title(f'Loss - Fase {phase}', fontsize=14)
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history.history:
            axes[2].plot(history.history['auc'], label='Treino', linewidth=2)
            axes[2].plot(history.history['val_auc'], label='Validação', linewidth=2)
            axes[2].set_title(f'AUC - Fase {phase}', fontsize=14)
            axes[2].set_xlabel('Época')
            axes[2].set_ylabel('AUC')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / f'training_curves_phase_{phase}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráficos de treinamento salvos: training_curves_phase_{phase}.png")
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                             title: str = 'Matriz de Confusão',
                             filename: str = 'confusion_matrix.png'):
        """Plota e salva matriz de confusão"""
        plt.figure(figsize=(10, 8))
        
        # Normaliza para porcentagens
        conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        # Cria anotações com valores absolutos e porcentagens
        annotations = np.empty_like(conf_matrix).astype(str)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                annotations[i, j] = f'{conf_matrix[i, j]}\n({conf_matrix_pct[i, j]:.1f}%)'
        
        sns.heatmap(
            conf_matrix,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Número de amostras'}
        )
        
        plt.title(title, fontsize=16)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.run_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matriz de confusão salva: {filename}")
    
    def _plot_roc_curves(self, true_labels: np.ndarray, 
                        predictions: np.ndarray):
        """Plota curvas ROC para cada classe"""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        # Para cada classe
        for i in range(self.num_classes):
            # Binariza para one-vs-rest
            y_true_binary = (true_labels == i).astype(int)
            y_score = predictions[:, i]
            
            # Calcula ROC
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plota
            plt.plot(
                fpr, tpr,
                label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Linha de referência
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        plt.title('Curvas ROC - Classificação ALN', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Curvas ROC salvas: roc_curves.png")
    
    def _save_evaluation_results(self, results: Dict):
        """Salva resultados da avaliação em arquivos"""
        # Salva métricas em JSON
        metrics = {
            'test_loss': float(results['test_loss']),
            'test_accuracy': float(results['test_accuracy']),
            'test_auc': float(results['test_auc']),
            'per_class_metrics': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            metrics['per_class_metrics'][class_name] = {
                'precision': float(results['precision'][i]),
                'recall': float(results['recall'][i]),
                'f1_score': float(results['f1_score'][i]),
                'support': int(results['support'][i]),
                'roc_auc': float(results['roc_auc_scores'][class_name])
            }
        
        with open(self.run_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Salva classification report
        report_df = pd.DataFrame(results['classification_report']).transpose()
        report_df.to_csv(self.run_dir / 'classification_report.csv')
        
        # Salva predições
        predictions_df = pd.DataFrame({
            'true_label': results['true_labels'],
            'predicted_label': results['predicted_labels'],
            'confidence': np.max(results['predictions'], axis=1)
        })
        
        # Adiciona probabilidades para cada classe
        for i, class_name in enumerate(self.class_names):
            predictions_df[f'prob_{class_name}'] = results['predictions'][:, i]
        
        predictions_df.to_csv(self.run_dir / 'test_predictions.csv', index=False)
        
        print("💾 Resultados salvos em:")
        print(f"  - test_metrics.json")
        print(f"  - classification_report.csv")
        print(f"  - test_predictions.csv")
    
    def generate_final_report(self, history: tf.keras.callbacks.History,
                            test_results: Dict,
                            patient_results: Dict = None):
        """
        Gera relatório final em formato HTML
        """
        from datetime import datetime
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Treinamento - Inception V3 ALN</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .metric-box {{
                    display: inline-block;
                    padding: 20px;
                    margin: 10px;
                    background-color: #e8f4f8;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2196F3;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Relatório de Treinamento - Inception V3 para Classificação ALN</h1>
                <p><strong>Data:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <p><strong>Diretório:</strong> {self.run_dir}</p>
                
                <h2>📊 Métricas Gerais</h2>
                <div>
                    <div class="metric-box">
                        <div>Acurácia</div>
                        <div class="metric-value">{test_results['accuracy']:.2%}</div>
                    </div>
                    <div class="metric-box">
                        <div>AUC Médio</div>
                        <div class="metric-value">{test_results['test_auc']:.3f}</div>
                    </div>
                    <div class="metric-box">
                        <div>Loss</div>
                        <div class="metric-value">{test_results['test_loss']:.4f}</div>
                    </div>
                </div>
                
                <h2>📈 Métricas por Classe</h2>
                <table>
                    <tr>
                        <th>Classe</th>
                        <th>Precisão</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Suporte</th>
                        <th>ROC AUC</th>
                    </tr>
        """
        
        for i, class_name in enumerate(self.class_names):
            html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{test_results['precision'][i]:.3f}</td>
                        <td>{test_results['recall'][i]:.3f}</td>
                        <td>{test_results['f1_score'][i]:.3f}</td>
                        <td>{test_results['support'][i]}</td>
                        <td>{test_results['roc_auc_scores'][class_name]:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>📊 Visualizações</h2>
                <h3>Curvas de Treinamento</h3>
                <img src="training_curves_phase_combined.png" alt="Curvas de Treinamento">
                
                <h3>Matriz de Confusão</h3>
                <img src="confusion_matrix.png" alt="Matriz de Confusão">
                
                <h3>Curvas ROC</h3>
                <img src="roc_curves.png" alt="Curvas ROC">
        """
        
        if patient_results:
            html_content += f"""
                <h2>👥 Resultados por Paciente</h2>
                <p><strong>Acurácia por Paciente:</strong> {patient_results['accuracy']:.2%}</p>
                <img src="confusion_matrix_patient_mean.png" alt="Matriz de Confusão por Paciente">
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Salva HTML
        with open(self.run_dir / 'relatorio_final.html', 'w') as f:
            f.write(html_content)
        
        print(f"\n📋 Relatório final salvo: {self.run_dir / 'relatorio_final.html'}")


def main():
    """
    Função principal para executar o treinamento
    """
    print("="*80)
    print("🧠 TREINAMENTO INCEPTION V3 - CLASSIFICAÇÃO ALN")
    print("="*80)
    
    # Configurações
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    # Verifica se arquivos existem
    if not patches_dir.exists():
        print(f"❌ Erro: Diretório de patches não encontrado: {patches_dir}")
        return
    
    if not clinical_data_path.exists():
        print(f"❌ Erro: Arquivo CSV não encontrado: {clinical_data_path}")
        return
    
    # Inicializa classificador
    classifier = InceptionV3ALNClassifier(
        patches_dir=str(patches_dir),
        clinical_data_path=str(clinical_data_path)
    )
    
    # Carrega dados clínicos
    classifier.load_clinical_data()
    
    # Prepara dataset
    datasets = classifier.prepare_dataset(
        max_patches_per_patient=10,  # Limite para desenvolvimento rápido
        test_size=0.2,               # 20% para teste
        val_size=0.1                 # 10% para validação
    )
    
    # Constrói modelo
    model = classifier.build_model(
        learning_rate=0.001,
        dropout_rate=0.5
    )
    
    # Treina modelo
    history = classifier.train_model(
        datasets=datasets,
        epochs=30,                   # Épocas para primeira fase
        batch_size=16,
        fine_tune=True,              # Ativa fine-tuning
        fine_tune_epochs=15,         # Épocas para fine-tuning
        fine_tune_layers=100         # Camadas a descongelar
    )
    
    # Avalia no conjunto de teste
    test_results = classifier.evaluate_model(
        test_df=datasets['test'],
        batch_size=16
    )
    
    # Avalia por paciente
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
    
    print("\n" + "="*80)
    print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Todos os resultados salvos em: {classifier.run_dir}")
    print("\n📊 Resumo Final:")
    print(f"  - Acurácia no teste: {test_results['accuracy']:.4f}")
    print(f"  - Acurácia por paciente: {patient_results['accuracy']:.4f}")
    print(f"  - AUC médio: {test_results['test_auc']:.4f}")
    

if __name__ == "__main__":
    # Configura TensorFlow para usar menos memória
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Executa treinamento
    main()
