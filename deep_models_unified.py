"""
Pipeline unificado para classificação de metástase ALN usando modelos profundos.
Combina Inception V3 e MobileNet V2 com funções compartilhadas.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional

# Configura GPU se disponível
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_memory_limit(physical_devices[0], 5500)
    print(f"GPU disponível: {physical_devices[0]}")
else:
    print("Executando em CPU")


class BaseALNClassifier:
    """
    Classe base com funcionalidades compartilhadas entre Inception e MobileNet
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        """
        Inicializa o classificador base
        
        Args:
            patches_dir: Diretório contendo patches organizados por paciente
            clinical_data_path: Caminho para arquivo CSV com dados clínicos
        """
        self.patches_dir = Path(patches_dir)
        self.clinical_data_path = clinical_data_path
        self.clinical_data = None
        self.model = None
        self.class_weights = None
        self.num_classes = 3
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        
        # Será definido pelas subclasses
        self.input_size = None
        self.model_name = None
        
        # Carrega dados clínicos
        self._load_clinical_data()
        
    def _load_clinical_data(self):
        """Carrega e processa dados clínicos"""
        print("Carregando dados clínicos...")
        self.clinical_data = pd.read_csv(self.clinical_data_path)
        
        # Mapeia ALN status para códigos numéricos
        aln_mapping = {
            'N0': 0,
            'N+(1-2)': 1, 
            'N+(>2)': 2
        }
        self.clinical_data['ALN_encoded'] = self.clinical_data['ALN status'].map(aln_mapping)
        
        print(f"Dados carregados: {len(self.clinical_data)} pacientes")
        print(f"Distribuição das classes: {self.clinical_data['ALN status'].value_counts().to_dict()}")

    def prepare_dataset(self, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Prepara dataset com divisão correta por paciente
        SEMPRE USA TODOS OS PATCHES DISPONÍVEIS

        Args:
            test_size: Proporção para teste (padrão 20%)

        Returns:
            Dicionário com DataFrames para treino, validação e teste
        """
        print(f"\n📁 Preparando dataset completo de patches para {self.model_name}...")
        print(f"  Usando TODOS os patches disponíveis")
        print(f"  Tamanho de entrada: {self.input_size}")

        # Coleta informações de TODOS os patches
        patch_data = []
        patient_patch_counts = {}

        for _, patient_row in self.clinical_data.iterrows():
            patient_id = str(patient_row['Patient ID'])
            aln_class = int(patient_row['ALN_encoded'])
            patient_dir = self.patches_dir / patient_id

            if not patient_dir.exists():
                continue

            # Lista TODOS os patches do paciente
            patch_files = sorted(list(patient_dir.glob("*.jpg")))

            if not patch_files:
                continue

            patient_patch_counts[patient_id] = len(patch_files)

            # Adiciona TODOS os patches ao dataset
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
        print(f"  Máximo patches/paciente: {np.max(list(patient_patch_counts.values()))}")
        print(f"  Mínimo patches/paciente: {np.min(list(patient_patch_counts.values()))}")

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

        # Segunda divisão: 75% treino, 25% validação (do conjunto treino+val)
        patients_train, patients_val, y_train, y_val = train_test_split(
            patients_train_val,
            y_train_val,
            test_size=0.25,
            random_state=43,
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

        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'all': df_patches
        }

    def create_data_generator(self, df_patches, batch_size, shuffle=True, augment=False):
        """
        Cria gerador de dados para treinamento
        
        Args:
            df_patches: DataFrame com informações dos patches
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar os dados
            augment: Se deve aplicar data augmentation
            
        Yields:
            Batches de (imagens, labels)
        """
        # Configuração de data augmentation para histologia
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.1,
                fill_mode='nearest',
                preprocessing_function=None
            )
        else:
            datagen = ImageDataGenerator(preprocessing_function=None)
            
        while True:
            if shuffle:
                df_shuffled = df_patches.sample(frac=1).reset_index(drop=True)
            else:
                df_shuffled = df_patches
                
            for i in range(0, len(df_shuffled), batch_size):
                batch_df = df_shuffled.iloc[i:i+batch_size]
                
                batch_images = []
                batch_labels = []
                
                for _, row in batch_df.iterrows():
                    # Carrega imagem
                    img = load_img(row['patch_path'], target_size=self.input_size)
                    img = img_to_array(img)
                    
                    # Aplica augmentation se especificado
                    if augment:
                        img = datagen.random_transform(img)
                    
                    # Aplica pré-processamento específico do modelo
                    img = np.expand_dims(img, axis=0)
                    img = self.preprocess_input(img)
                    img = np.squeeze(img, axis=0)
                        
                    batch_images.append(img)
                    
                    # One-hot encoding da label
                    label = np.zeros(self.num_classes)
                    label[int(row['class'])] = 1
                    batch_labels.append(label)
                    
                yield np.array(batch_images), np.array(batch_labels)
                
    def calculate_class_weights(self, train_df):
        """Calcula pesos das classes para balanceamento"""
        y_train = train_df['class'].values
        classes = np.unique(y_train)
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        self.class_weights = dict(zip(classes, weights))
        print(f"Pesos das classes calculados: {self.class_weights}")
        
        return self.class_weights
        
    def train_model(self, datasets, epochs=50, fine_tune_epochs=15):
        """
        Treina o modelo com estratégia de two-stage training
        
        Args:
            datasets: Dicionário com datasets de treino, validação e teste
            epochs: Número de épocas para feature extraction
            fine_tune_epochs: Número de épocas para fine-tuning
            
        Returns:
            História do treinamento
        """
        train_df = datasets['train']
        val_df = datasets['val']
        
        # Calcula pesos das classes
        self.calculate_class_weights(train_df)
        
        # Define batch size baseado no modelo
        batch_size = self.get_optimal_batch_size()
        
        # Cria geradores
        train_generator = self.create_data_generator(
            train_df, 
            batch_size=batch_size, 
            shuffle=True, 
            augment=True
        )
        
        val_generator = self.create_data_generator(
            val_df, 
            batch_size=batch_size, 
            shuffle=False, 
            augment=False
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.model_name.lower()}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calcula steps
        steps_per_epoch = len(train_df) // batch_size
        validation_steps = len(val_df) // batch_size
        
        print(f"\n🚀 Iniciando treinamento {self.model_name}...")
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Steps por época: {steps_per_epoch}")
        print(f"📊 Validation steps: {validation_steps}")
        
        # Stage 1: Feature extraction
        print(f"\n📌 Stage 1: Feature Extraction ({epochs} épocas)")
        
        # Reconstrói modelo sem fine-tuning
        self.build_model(fine_tune=False)
        
        history_fe = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Stage 2: Fine-tuning
        if fine_tune_epochs > 0:
            print(f"\n📌 Stage 2: Fine-tuning ({fine_tune_epochs} épocas)")
            
            # Salva pesos e reconstrói com fine-tuning
            weights = self.model.get_weights()
            self.build_model(fine_tune=True)
            self.model.set_weights(weights)
            
            # Callbacks com learning rate menor
            ft_callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-8,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'{self.model_name.lower()}_finetuned.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            history_ft = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=fine_tune_epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=ft_callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            # Combina histórias
            history = {
                'loss': history_fe.history['loss'] + history_ft.history['loss'],
                'accuracy': history_fe.history['accuracy'] + history_ft.history['accuracy'],
                'val_loss': history_fe.history['val_loss'] + history_ft.history['val_loss'],
                'val_accuracy': history_fe.history['val_accuracy'] + history_ft.history['val_accuracy']
            }
        else:
            history = history_fe.history
            
        print("\n✅ Treinamento concluído!")
        
        return history
        
    def evaluate_model(self, test_df, batch_size=None):
        """
        Avalia modelo no conjunto de teste
        
        Args:
            test_df: DataFrame com dados de teste
            batch_size: Tamanho do batch (None = usa padrão do modelo)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n📊 Avaliando modelo no conjunto de teste...")
        
        if batch_size is None:
            batch_size = self.get_optimal_batch_size()
        
        # Cria gerador sem augmentation
        test_generator = self.create_data_generator(
            test_df,
            batch_size=batch_size,
            shuffle=False,
            augment=False
        )
        
        # Predições
        steps = len(test_df) // batch_size + (1 if len(test_df) % batch_size else 0)
        
        y_true = []
        y_pred = []
        
        for i in range(steps):
            batch_x, batch_y = next(test_generator)
            predictions = self.model.predict(batch_x, verbose=0)
            
            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
            
            if len(y_true) >= len(test_df):
                y_true = y_true[:len(test_df)]
                y_pred = y_pred[:len(test_df)]
                break
        
        # Métricas
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print(f"\n✅ Acurácia no teste: {accuracy:.4f}")
        print("\n📋 Relatório de classificação:")
        print(classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names
        ))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
    def save_model(self, filepath):
        """Salva modelo treinado"""
        self.model.save(filepath)
        print(f"✅ Modelo salvo em: {filepath}")
        
    def plot_training_history(self, history):
        """Plota histórico de treinamento"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Treino')
        plt.plot(history['val_accuracy'], label='Validação')
        plt.title(f'Acurácia do Modelo {self.model_name}')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Treino')
        plt.plot(history['val_loss'], label='Validação')
        plt.title(f'Loss do Modelo {self.model_name}')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Matriz de Confusão - {self.model_name}')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        plt.savefig(f'{self.model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


class InceptionV3Classifier(BaseALNClassifier):
    """
    Classificador usando Inception V3
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        super().__init__(patches_dir, clinical_data_path)
        self.input_size = (299, 299)
        self.model_name = "InceptionV3"
        self.preprocess_input = preprocess_inception
        
    def build_model(self, fine_tune=True):
        """Constrói modelo Inception V3"""
        print(f"Construindo modelo {self.model_name}...")
        
        # Modelo base
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Camadas de classificação
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Fine-tuning
        if fine_tune:
            # Congela primeiras 100 camadas
            for layer in base_model.layers[:100]:
                layer.trainable = False
            for layer in base_model.layers[100:]:
                layer.trainable = True
            optimizer = Adam(learning_rate=0.0001)
        else:
            # Congela todo modelo base
            for layer in base_model.layers:
                layer.trainable = False
            optimizer = Adam(learning_rate=0.001)
            
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modelo construído! Total de parâmetros: {self.model.count_params():,}")
        
    def get_optimal_batch_size(self):
        """Retorna batch size otimizado para Inception"""
        return 16  # Inception é pesado, batch menor


class MobileNetV2Classifier(BaseALNClassifier):
    """
    Classificador usando MobileNet V2
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        super().__init__(patches_dir, clinical_data_path)
        self.input_size = (224, 224)
        self.model_name = "MobileNetV2"
        self.preprocess_input = preprocess_mobilenet
        
    def build_model(self, fine_tune=True):
        """Constrói modelo MobileNet V2"""
        print(f"Construindo modelo {self.model_name}...")
        
        # Modelo base
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            alpha=1.0
        )
        
        # Camadas de classificação
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.3, name='dropout2')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Fine-tuning
        if fine_tune:
            # MobileNet tem menos camadas, congela primeiras 80
            for layer in base_model.layers[:80]:
                layer.trainable = False
            for layer in base_model.layers[80:]:
                layer.trainable = True
            optimizer = Adam(learning_rate=0.0001)
        else:
            # Congela todo modelo base
            for layer in base_model.layers:
                layer.trainable = False
            optimizer = Adam(learning_rate=0.001)
            
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Modelo construído! Total de parâmetros: {self.model.count_params():,}")
        
    def get_optimal_batch_size(self):
        """Retorna batch size otimizado para MobileNet"""
        return 32  # MobileNet é leve, batch maior


def train_deep_model(model_type="inception"):
    """
    Função principal para treinar modelo profundo
    
    Args:
        model_type: "inception" ou "mobilenet"
    """
    # Define caminhos
    base_dir = Path(__file__).parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"
    
    # Verifica arquivos
    if not patches_dir.exists():
        print(f"❌ Diretório de patches não encontrado: {patches_dir}")
        return
        
    if not clinical_data_path.exists():
        print(f"❌ Arquivo CSV não encontrado: {clinical_data_path}")
        return
    
    # Cria classificador apropriado
    if model_type.lower() == "inception":
        classifier = InceptionV3Classifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        epochs_fe = 40  # Feature extraction
        epochs_ft = 15  # Fine-tuning
    else:  # mobilenet
        classifier = MobileNetV2Classifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        epochs_fe = 35  # Feature extraction
        epochs_ft = 15  # Fine-tuning
    
    print(f"\n{'='*80}")
    print(f"🚀 TREINAMENTO {classifier.model_name.upper()} - DATASET COMPLETO")
    print(f"{'='*80}")
    
    # Prepara dataset completo
    print("\n📊 PREPARAÇÃO DOS DADOS")
    datasets = classifier.prepare_dataset(test_size=0.2)
    
    # Constrói e treina modelo
    print(f"\n🏗️ CONSTRUÇÃO E TREINAMENTO DO MODELO")
    history = classifier.train_model(
        datasets=datasets,
        epochs=epochs_fe,
        fine_tune_epochs=epochs_ft
    )
    
    # Avalia modelo
    print("\n📈 AVALIAÇÃO FINAL")
    results = classifier.evaluate_model(datasets['test'])
    
    # Visualizações
    print("\n📊 GERANDO VISUALIZAÇÕES")
    classifier.plot_training_history(history)
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # Salva modelo final
    model_path = base_dir / f"{classifier.model_name.lower()}_aln_final.h5"
    classifier.save_model(str(model_path))
    
    # Relatório final
    print(f"\n{'='*60}")
    print("📋 RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"✅ Modelo: {classifier.model_name}")
    print(f"🎯 Acurácia final: {results['accuracy']:.4f}")
    print(f"💾 Modelo salvo em: {model_path}")
    
    # Métricas por classe
    report = results['classification_report']
    print("\n📊 Métricas por classe:")
    for class_name in classifier.class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"  {class_name}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, "
                  f"F1={metrics['f1-score']:.3f}, "
                  f"N={metrics['support']}")
    
    # Estatísticas do dataset
    print(f"\n📈 Estatísticas do treinamento:")
    print(f"  Total de patches: {len(datasets['all'])}")
    print(f"  Patches treino: {len(datasets['train'])}")
    print(f"  Patches validação: {len(datasets['val'])}")
    print(f"  Patches teste: {len(datasets['test'])}")
    print(f"  Épocas totais: {epochs_fe + epochs_ft}")
    
    return classifier, history, results


def main():
    """Função principal com interface simples"""
    print("\n" + "="*80)
    print("🧠 CLASSIFICAÇÃO DE METÁSTASE ALN - MODELOS PROFUNDOS")
    print("="*80)
    print("\nEscolha o modelo para treinar com o dataset COMPLETO:")
    print("1. Inception V3 (máxima acurácia, ~23.8M parâmetros)")
    print("2. MobileNet V2 (eficiente, ~3.5M parâmetros)")
    print("\n0. Sair")
    
    choice = input("\n👉 Digite sua escolha (0-2): ").strip()
    
    if choice == "1":
        print("\n🎯 Iniciando treinamento com INCEPTION V3...")
        print("⏱️ Tempo estimado: 3-5 horas (dependendo da GPU)")
        confirm = input("\nConfirmar? (s/N): ")
        if confirm.lower() == 's':
            train_deep_model("inception")
        else:
            print("❌ Treinamento cancelado")
            
    elif choice == "2":
        print("\n🚀 Iniciando treinamento com MOBILENET V2...")
        print("⏱️ Tempo estimado: 1-2 horas (dependendo da GPU)")
        confirm = input("\nConfirmar? (s/N): ")
        if confirm.lower() == 's':
            train_deep_model("mobilenet")
        else:
            print("❌ Treinamento cancelado")
            
    elif choice == "0":
        print("\n👋 Saindo...")
        return
        
    else:
        print("\n❌ Opção inválida!")
        return
    
    print("\n🎉 Processo concluído!")


if __name__ == "__main__":
    main()
