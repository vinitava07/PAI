import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from inception import InceptionV4

class InceptionALNClassifier:
    """
    Pipeline completa para classificação de patches de câncer de mama usando Inception V4
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
        self.input_size = (299, 299)  # Tamanho padrão Inception V4
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
        
    def prepare_dataset(self, max_patches_per_patient=None, test_size=0.2, val_size=0.1):
        """
        Prepara dataset de patches com suas respectivas labels
        
        Args:
            max_patches_per_patient: Máximo de patches por paciente (None = todos)
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            
        Returns:
            Dicionário com datasets preparados
        """
        print("Preparando dataset de patches...")
        
        # Lista para armazenar dados dos patches
        patch_data = []
        
        # Itera sobre pacientes
        for _, patient_row in self.clinical_data.iterrows():
            patient_id = str(patient_row['Patient ID'])
            aln_class = patient_row['ALN_encoded']
            
            if pd.isna(aln_class):
                continue
                
            patient_dir = self.patches_dir / patient_id
            
            if not patient_dir.exists():
                continue
                
            # Lista patches do paciente
            patch_files = list(patient_dir.glob("*.jpg"))
            
            # Limita número de patches se especificado
            if max_patches_per_patient and len(patch_files) > max_patches_per_patient:
                patch_files = patch_files[:max_patches_per_patient]
                
            # Adiciona cada patch à lista
            for patch_file in patch_files:
                patch_data.append({
                    'patch_path': str(patch_file),
                    'patient_id': patient_id,
                    'class': int(aln_class),
                    'class_name': self.class_names[int(aln_class)]
                })
        
        if not patch_data:
            raise ValueError("Nenhum patch encontrado!")
            
        # Converte para DataFrame
        df_patches = pd.DataFrame(patch_data)
        
        print(f"Total de patches: {len(df_patches)}")
        print(f"Distribuição por classe: {df_patches['class_name'].value_counts().to_dict()}")
        
        # Divide por paciente (não por patch) para evitar data leakage
        unique_patients = df_patches['patient_id'].unique()
        
        # Estratifica por distribuição de classes dos pacientes
        patient_classes = []
        for pid in unique_patients:
            patient_class = df_patches[df_patches['patient_id'] == pid]['class'].iloc[0]
            patient_classes.append(patient_class)
            
        # Split de pacientes
        train_patients, temp_patients, _, temp_classes = train_test_split(
            unique_patients, patient_classes, 
            test_size=(test_size + val_size), 
            random_state=42, 
            stratify=patient_classes
        )
        
        # Split validação e teste
        val_patients, test_patients = train_test_split(
            temp_patients, 
            test_size=(test_size / (test_size + val_size)), 
            random_state=42,
            stratify=temp_classes
        )
        
        # Separa patches por conjunto
        train_patches = df_patches[df_patches['patient_id'].isin(train_patients)]
        val_patches = df_patches[df_patches['patient_id'].isin(val_patients)]
        test_patches = df_patches[df_patches['patient_id'].isin(test_patients)]
        
        print(f"Patches treino: {len(train_patches)}")
        print(f"Patches validação: {len(val_patches)}")
        print(f"Patches teste: {len(test_patches)}")
        
        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'all': df_patches
        }
        
    def preprocess_image(self, image_path):
        """
        Pré-processa uma imagem para o modelo Inception V4
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Array numpy da imagem processada
        """
        # Carrega imagem
        img = load_img(image_path, target_size=self.input_size)
        img_array = img_to_array(img)
        
        # Normaliza para [0, 1]
        img_array = img_array / 255.0
        
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
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()
            
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
                    # Carrega e processa imagem
                    img = self.preprocess_image(row['patch_path'])
                    
                    # Aplica augmentation se especificado
                    if augment:
                        img = datagen.random_transform(img)
                        
                    batch_images.append(img)
                    
                    # One-hot encoding da label
                    label = np.zeros(self.num_classes)
                    label[int(row['class'])] = 1
                    batch_labels.append(label)
                    
                yield np.array(batch_images), np.array(batch_labels)
                
    def build_model(self):
        """Constrói e configura modelo Inception V4 para 3 classes"""
        print("Construindo modelo Inception V4...")
        
        # Cria modelo base
        base_model = InceptionV4()
        
        # Remove última camada (1000 classes) e adiciona nova para 3 classes
        x = base_model.layers[-3].output  # Antes da camada Dense final
        x = Dense(units=512, activation='relu', name='new_dense_1')(x)
        x = Dropout(rate=0.5, name='new_dropout')(x)
        predictions = Dense(units=self.num_classes, activation='softmax', name='aln_predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compila modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Modelo construído com sucesso!")
        print(f"Total de parâmetros: {self.model.count_params():,}")
        
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
        
    def train_model(self, datasets, epochs=50, batch_size=16):
        """
        Treina o modelo
        
        Args:
            datasets: Dicionário com datasets de treino, validação e teste
            epochs: Número de épocas
            batch_size: Tamanho do batch
            
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
            ModelCheckpoint(
                'best_inception_aln_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinamento
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Treinamento concluído!")
        return history
        
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
        ax1.set_title('Acurácia por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Treino')
        ax2.plot(history.history['val_loss'], label='Validação')
        ax2.set_title('Loss por Época')
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
        plt.title('Matriz de Confusão - Inception V4')
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
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
