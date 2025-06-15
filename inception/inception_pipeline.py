import pandas as pd
import numpy as np
from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List, Optional

class InceptionV3ALNClassifier:
    """
    Pipeline completa para classificação de patches de câncer de mama usando Inception V3 pré-treinado
    """
    
    def __init__(self, patches_dir, clinical_data_path):
        """
        Inicializa o classificador
        
        Args:
            patches_dir: Diretório contendo patches organizados por paciente
            clinical_data_path: Caminho para arquivo CSV com dados clínicos
        """
        self.patches_dir = Path(patches_dir)
        self.clinical_data_path = clinical_data_path
        self.clinical_data = None
        self.model = None
        self.class_weights = None
        self.input_size = (299, 299)  # Tamanho padrão Inception V3
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

        # Salva informações da divisão
        split_info = {
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

        # with open(self.run_dir / 'dataset_split_info.json', 'w') as f:
        #     json.dump(split_info, f, indent=2)

        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'all': df_patches
        }

    def preprocess_image(self, image_path):
        """
        Pré-processa uma imagem para o modelo Inception V3
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Array numpy da imagem processada
        """
        # Carrega imagem
        img = load_img(image_path, target_size=self.input_size)
        img_array = img_to_array(img)
        
        # Adiciona dimensão do batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Pré-processamento específico do Inception V3 (normalização [-1, 1])
        img_array = preprocess_input(img_array)
        
        # Remove dimensão do batch
        img_array = np.squeeze(img_array, axis=0)
        
        return img_array
        
    def create_data_generator(self, df_patches, batch_size=16, shuffle=True, augment=False):
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
        # Configuração de data augmentation
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.1,
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
                    
                    # Aplica pré-processamento do Inception V3
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
        Constrói modelo Inception V3 pré-treinado para 3 classes
        
        Args:
            fine_tune: Se deve fazer fine-tuning das camadas
            freeze_layers: Número de camadas a congelar (do início)
        """
        print("Construindo modelo Inception V3 pré-treinado...")
        
        # Cria modelo base pré-treinado (sem as camadas de classificação)
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Adiciona camadas de classificação personalizada
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='dropout1')(x)
        # x = Dense(512, activation='relu', name='fc2')(x)
        # x = Dropout(0.3, name='dropout2')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='aln_predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Configuração de fine-tuning
        if fine_tune:
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
        
        # Compila modelo
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
        trainable_params = sum([np.prod(layer.get_weights()[0].shape) 
                               for layer in self.model.layers 
                               if layer.trainable and layer.get_weights()])
        print(f"Parâmetros treináveis: {trainable_params:,}")
        
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
        
    def train_model(self, datasets, epochs=50, batch_size=16, fine_tune_epochs=False):
        """
        Treina o modelo com estratégia de two-stage training
        
        Args:
            datasets: Dicionário com datasets de treino, validação e teste
            epochs: Número de épocas para feature extraction
            batch_size: Tamanho do batch
            fine_tune_epochs: Número de épocas para fine-tuning (None = não faz)
            
        Returns:
            Histórico de treinamento
        """
        print("Iniciando treinamento...")
        
        train_df = datasets['train']
        val_df = datasets['val']
        
        # Calcula pesos das classes
        class_weights = self.calculate_class_weights(train_df)
        
        # Calcula steps por época
        steps_per_epoch = len(train_df) // batch_size
        validation_steps = len(val_df) // batch_size
        
        # Geradores de dados
        train_generator = self.create_data_generator(
            train_df, batch_size=batch_size, shuffle=True, augment=True
        )
        val_generator = self.create_data_generator(
            val_df, batch_size=batch_size, shuffle=False, augment=False
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_inception_v3_aln_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Fase 1: Feature extraction (modelo base congelado)
        print("\n" + "="*50)
        print("FASE 1: FEATURE EXTRACTION")
        print("="*50)
        
        history1 = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fase 2: Fine-tuning (se especificado)
        if fine_tune_epochs:
            print("\n" + "="*50)
            print("FASE 2: FINE-TUNING")
            print("="*50)
            
            # Descongela mais camadas para fine-tuning
            for layer in self.model.layers[-50:]:  # Últimas 50 camadas
                layer.trainable = True
            
            # Recompila com learning rate menor
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),  # LR muito baixo
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks para fine-tuning
            callbacks_ft = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=5,
                    min_lr=1e-8,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_inception_v3_aln_finetuned.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            history2 = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=fine_tune_epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                class_weight=class_weights,
                callbacks=callbacks_ft,
                verbose=1
            )
            
            # Combina históricos
            for key in history1.history.keys():
                history1.history[key].extend(history2.history[key])
        
        print("Treinamento concluído!")
        return history1
        
    def evaluate_model(self, test_df, batch_size=16):
        """
        Avalia modelo no conjunto de teste
        
        Args:
            test_df: DataFrame com dados de teste
            batch_size: Tamanho do batch
            
        Returns:
            Dicionário com resultados da avaliação
        """
        print("Avaliando modelo...")
        
        # Prepara dados de teste
        test_images = []
        test_labels = []
        
        for _, row in test_df.iterrows():
            img = self.preprocess_image(row['patch_path'])
            test_images.append(img)
            test_labels.append(int(row['class']))
            
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        # Predições
        predictions = self.model.predict(test_images, batch_size=batch_size, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Métricas
        accuracy = np.mean(predicted_classes == test_labels)
        
        # Relatório detalhado
        report = classification_report(
            test_labels, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Matriz de confusão
        conf_matrix = confusion_matrix(test_labels, predicted_classes)
        
        results = {
            'accuracy': accuracy,
            'predictions': predicted_classes,
            'true_labels': test_labels,
            'probabilities': predictions,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        # Exibe resultados
        print(f"\nAcurácia no teste: {accuracy:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(test_labels, predicted_classes, target_names=self.class_names))
        
        return results
        
    def plot_training_history(self, history):
        """Plota histórico de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Acurácia
        ax1.plot(history.history['accuracy'], label='Treino')
        ax1.plot(history.history['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia por Época - Inception V3')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Treino')
        ax2.plot(history.history['val_loss'], label='Validação')
        ax2.set_title('Loss por Época - Inception V3')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, conf_matrix):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Matriz de Confusão - Inception V3')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath):
        """Salva modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath):
        """Carrega modelo salvo"""
        from tensorflow.keras.models import load_model # type: ignore
        self.model = load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
        
    def predict_patient(self, patient_patches, aggregate='mean'):
        """
        Faz predição para um paciente baseado em todos seus patches
        
        Args:
            patient_patches: Lista de caminhos de patches do paciente
            aggregate: Método de agregação ('mean', 'max', 'vote')
            
        Returns:
            Predição final do paciente
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
            
        # Processa todos os patches
        patch_predictions = []
        
        for patch_path in patient_patches:
            img = self.preprocess_image(patch_path)
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict(img, verbose=0)
            patch_predictions.append(pred[0])
            
        patch_predictions = np.array(patch_predictions)
        
        # Agregação
        if aggregate == 'mean':
            final_pred = np.mean(patch_predictions, axis=0)
        elif aggregate == 'max':
            final_pred = np.max(patch_predictions, axis=0)
        elif aggregate == 'vote':
            # Voto majoritário
            patch_classes = np.argmax(patch_predictions, axis=1)
            final_class = np.bincount(patch_classes).argmax()
            final_pred = np.zeros(self.num_classes)
            final_pred[final_class] = 1.0
        else:
            raise ValueError("Método de agregação inválido")
            
        return final_pred, patch_predictions