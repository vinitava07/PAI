import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Usando MobileNetV2 ao invés de InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Pré-processamento específico do MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class MobileNetALNClassifier:
    """
    Pipeline completa para classificação de patches de câncer de mama usando MobileNetV2 pré-treinado
    Adaptado do InceptionV3ALNClassifier para usar MobileNetV2
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        """
        Inicializa o classificador MobileNet
        
        Args:
            patches_dir: Diretório contendo patches organizados por paciente
            clinical_data_path: Caminho para arquivo CSV com dados clínicos
        """
        self.patches_dir = Path(patches_dir)
        self.clinical_data_path = clinical_data_path
        self.clinical_data = None
        self.model = None
        self.class_weights = None
        self.input_size = (224, 224)  # Tamanho padrão MobileNetV2 (menor que Inception)
        self.num_classes = 3
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        
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

    def prepare_dataset(self, max_patches_per_patient: Optional[int] = None,
                        test_size: float = 0.2,
                        val_size: float = 0.25) -> Dict[str, pd.DataFrame]:
        """
        Prepara dataset com divisão correta por paciente
        Reutiliza a lógica do Inception mas com input_size específico do MobileNet

        Args:
            max_patches_per_patient: Limite de patches por paciente (None = todos)
            test_size: Proporção para teste (padrão 20%)
            val_size: Proporção para validação do treino

        Returns:
            Dicionário com DataFrames para treino, validação e teste
        """
        print(f"\n📁 Preparando dataset de patches para MobileNetV2...")
        print(f"  Max patches/paciente: {max_patches_per_patient or 'Todos'}")
        print(f"  Tamanho de entrada: {self.input_size}")

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
        patients_train, patients_val, y_train, y_val = train_test_split(
            patients_train_val,
            y_train_val,
            train_size=0.75,
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

    def preprocess_image(self, image_path):
        """
        Pré-processa uma imagem para o modelo MobileNetV2
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Array numpy da imagem processada
        """
        # Carrega imagem com tamanho específico do MobileNet
        img = load_img(image_path, target_size=self.input_size)
        img_array = img_to_array(img)
        
        # Adiciona dimensão do batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Pré-processamento específico do MobileNetV2 (normalização [-1, 1])
        img_array = preprocess_input(img_array)
        
        # Remove dimensão do batch
        img_array = np.squeeze(img_array, axis=0)
        
        return img_array
        
    def create_data_generator(self, df_patches, batch_size=32, shuffle=True, augment=False):
        """
        Cria gerador de dados para treinamento
        Nota: batch_size maior que Inception pois MobileNet é mais leve
        
        Args:
            df_patches: DataFrame com informações dos patches
            batch_size: Tamanho do batch (padrão 32 para MobileNet)
            shuffle: Se deve embaralhar os dados
            augment: Se deve aplicar data augmentation
            
        Yields:
            Batches de (imagens, labels)
        """
        # Configuração de data augmentation otimizada para patches histológicos
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=90,  # Rotação em qualquer ângulo
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.15,  # Um pouco mais de zoom para MobileNet
                fill_mode='nearest',
                preprocessing_function=None  # Não usar preprocessing aqui
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
                    # Carrega imagem sem pré-processamento
                    img = load_img(row['patch_path'], target_size=self.input_size)
                    img = img_to_array(img)
                    
                    # Aplica augmentation se especificado
                    if augment:
                        img = datagen.random_transform(img)
                    
                    # Aplica pré-processamento do MobileNetV2
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    img = np.squeeze(img, axis=0)
                        
                    batch_images.append(img)
                    
                    # One-hot encoding da label
                    label = np.zeros(self.num_classes)
                    label[int(row['class'])] = 1
                    batch_labels.append(label)
                    
                yield np.array(batch_images), np.array(batch_labels)
                
    def build_model(self, fine_tune=True, freeze_layers=100):
        """
        Constrói modelo MobileNetV2 pré-treinado para 3 classes
        MobileNetV2 tem menos camadas que Inception, então ajustamos freeze_layers
        
        Args:
            fine_tune: Se deve fazer fine-tuning das camadas
            freeze_layers: Número de camadas a congelar (do início)
        """
        print("Construindo modelo MobileNetV2 pré-treinado...")
        
        # Cria modelo base pré-treinado (sem as camadas de classificação)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            alpha=1.0  # Largura padrão do modelo
        )
        
        # MobileNet tem menos camadas que Inception (~155 vs ~311)
        print(f"MobileNetV2 tem {len(base_model.layers)} camadas")
        
        # Adiciona camadas de classificação personalizada
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Camadas densas menores que Inception (MobileNet é mais eficiente)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.3, name='dropout2')(x)
        
        # Camada de saída
        predictions = Dense(self.num_classes, activation='softmax', name='aln_predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Configuração de fine-tuning
        if fine_tune:
            # MobileNet tem menos camadas, então ajustamos proporcionalmente
            freeze_layers = min(freeze_layers, 100)  # No máximo 100 camadas congeladas
            
            # Congela as primeiras camadas
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
            
            # Descongela as camadas finais
            for layer in base_model.layers[freeze_layers:]:
                layer.trainable = True
                
            print(f"Fine-tuning habilitado: {len(base_model.layers) - freeze_layers} camadas treináveis")
        else:
            # Congela todo o modelo base
            for layer in base_model.layers:
                layer.trainable = False
            print("Modelo base congelado - apenas camadas de classificação treináveis")
        
        # Compila modelo com otimizador apropriado
        if fine_tune:
            # Learning rate menor para fine-tuning
            optimizer = Adam(learning_rate=0.0001)
        else:
            # Learning rate normal para feature extraction
            optimizer = Adam(learning_rate=0.001)
            
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Modelo construído com sucesso!")
        print(f"Total de parâmetros: {self.model.count_params():,}")
        
        # Conta parâmetros treináveis
        trainable_count = np.sum([tf.keras.backend.count_params(w) 
                                 for w in self.model.trainable_weights])
        print(f"Parâmetros treináveis: {trainable_count:,}")
        
        return self.model
        
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
        print(f"Pesos das classes: {self.class_weights}")
        
        return self.class_weights
        
    def train_model(self, datasets, epochs=30, batch_size=32, fine_tune_epochs=10):
        """
        Treina o modelo com estratégia de two-stage training
        MobileNet treina mais rápido, então podemos usar mais épocas
        
        Args:
            datasets: Dicionário com datasets de treino, validação e teste
            epochs: Número de épocas para feature extraction
            batch_size: Tamanho do batch (maior para MobileNet)
            fine_tune_epochs: Número de épocas para fine-tuning
            
        Returns:
            História do treinamento
        """
        train_df = datasets['train']
        val_df = datasets['val']
        
        # Calcula pesos das classes
        self.calculate_class_weights(train_df)
        
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
        
        # Callbacks otimizados para MobileNet
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
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
                'mobilenet_v2_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calcula steps
        steps_per_epoch = len(train_df) // batch_size
        validation_steps = len(val_df) // batch_size
        
        print(f"\n🚀 Iniciando treinamento MobileNetV2...")
        print(f"📊 Steps por época: {steps_per_epoch}")
        print(f"📊 Validation steps: {validation_steps}")
        
        # Stage 1: Feature extraction (modelo base congelado)
        if epochs > 0:
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
        if fine_tune_epochs and fine_tune_epochs > 0:
            print(f"\n📌 Stage 2: Fine-tuning ({fine_tune_epochs} épocas)")
            
            # Reconstrói modelo com fine-tuning
            # Carrega pesos do stage 1
            if epochs > 0:
                weights = self.model.get_weights()
                self.build_model(fine_tune=True, freeze_layers=80)  # Menos camadas para MobileNet
                self.model.set_weights(weights)
            else:
                self.build_model(fine_tune=True, freeze_layers=80)
            
            # Callbacks com learning rate ainda menor
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
                    'mobilenet_v2_finetuned.h5',
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
            
            # Combina histórias se ambos stages foram executados
            if epochs > 0:
                history = {
                    'loss': history_fe.history['loss'] + history_ft.history['loss'],
                    'accuracy': history_fe.history['accuracy'] + history_ft.history['accuracy'],
                    'val_loss': history_fe.history['val_loss'] + history_ft.history['val_loss'],
                    'val_accuracy': history_fe.history['val_accuracy'] + history_ft.history['val_accuracy']
                }
            else:
                history = history_ft.history
        else:
            history = history_fe.history if epochs > 0 else None
            
        print("\n✅ Treinamento concluído!")
        
        return history
        
    def evaluate_model(self, test_df, batch_size=32):
        """
        Avalia modelo no conjunto de teste
        
        Args:
            test_df: DataFrame com dados de teste
            batch_size: Tamanho do batch
            
        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n📊 Avaliando modelo no conjunto de teste...")
        
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
            
            # Para garantir que não pegamos dados extras
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
        
    def predict(self, image_path):
        """
        Faz predição para uma única imagem
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Classe predita e probabilidades
        """
        # Pré-processa imagem
        img = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        
        # Predição
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        return {
            'class': self.class_names[predicted_class],
            'class_idx': predicted_class,
            'probabilities': predictions[0],
            'confidence': float(predictions[0][predicted_class])
        }
        
    def save_model(self, filepath):
        """Salva modelo treinado"""
        self.model.save(filepath)
        print(f"✅ Modelo salvo em: {filepath}")
        
    def load_model(self, filepath):
        """Carrega modelo salvo"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"✅ Modelo carregado de: {filepath}")
        
    def plot_training_history(self, history):
        """
        Plota histórico de treinamento
        
        Args:
            history: Histórico retornado pelo fit
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Treino')
        plt.plot(history['val_accuracy'], label='Validação')
        plt.title('Acurácia do Modelo MobileNetV2')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Treino')
        plt.plot(history['val_loss'], label='Validação')
        plt.title('Loss do Modelo MobileNetV2')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('mobilenet_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm):
        """
        Plota matriz de confusão
        
        Args:
            cm: Matriz de confusão
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Matriz de Confusão - MobileNetV2')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        plt.savefig('mobilenet_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
