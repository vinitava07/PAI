import os
import sys
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure # type: ignore
from skimage.feature import peak_local_max # type: ignore
from skimage.segmentation import watershed # type: ignore
import matplotlib.pyplot as plt
from skimage.color import rgb2hed # type: ignore
import pandas as pd
from pathlib import Path
import concurrent.futures
import multiprocessing
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, MobileNetV2 # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from datetime import datetime

# Configura GPU se disponível
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_memory_limit(physical_devices[0], 5500)
    print(f"GPU disponível: {physical_devices[0]}")
else:
    print("Executando em CPU")


base_dir = Path(__file__).parent.parent
patches_dir = base_dir / "patches"
clinical_data_path = base_dir / "patient-clinical-data.csv"
patches_certos = base_dir / "patches_certos.txt"

class HENucleusSegmentation:

    # Adicione esta função para filtrar melhor:
    def is_valid_nucleus(self, region):
        # Critérios mais rigorosos
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        return (
                15 <= region.area <= 400 and  # Área razoável
                circularity >= 0.3 and  # Não muito irregular
                region.eccentricity <= 0.94 and  # Não muito alongado
                region.solidity >= 0.1  # Não muito côncavo
        )

    def __init__(self):
        self.original_image = None
        self.hematoxylin_channel = None
        self.segmented_nuclei = None
        self.nucleus_features = []

    def preprocess_he_image(self, image_path):

        # print("Carregando e pré-processando imagem H&E...")

        # Carrega a imagem
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        # Converte BGR para RGB
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Deconvolução de cores para separar Hematoxilina e Eosina
        try:
            # Converte para espaço de cor HED (Hematoxylin-Eosin-DAB)
            hed = rgb2hed(image_rgb)

            # Canal 0 é Hematoxilina (núcleos)
            # Canal 1 é Eosina (citoplasma)
            # Canal 2 é DAB (não usado em H&E)

            # Extrai canal de hematoxilina
            self.hematoxylin_channel = hed[:, :, 0]

            # Normaliza para 0-255
            h_min = np.percentile(self.hematoxylin_channel, 1)
            h_max = np.percentile(self.hematoxylin_channel, 99)
            self.hematoxylin_channel = np.clip(self.hematoxylin_channel, h_min, h_max)
            self.hematoxylin_channel = ((self.hematoxylin_channel - h_min) /
                                       (h_max - h_min) * 255).astype(np.uint8)

        except:
            print("Usando método alternativo de separação de cores...")
            # Método alternativo: usar canal azul do espaço RGB
            # Núcleos são mais escuros no canal vermelho
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Inverte para que núcleos fiquem claros
            self.hematoxylin_channel = cv2.bitwise_not(gray)

            # Aumenta contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            self.hematoxylin_channel = clahe.apply(self.hematoxylin_channel)

        return self.hematoxylin_channel

    def segment_nuclei_he(self, hematoxylin_channel):

        # print("Segmentando núcleos...")

        # 1. Suavização para reduzir ruído
        smoothed = cv2.medianBlur(hematoxylin_channel, 3)

        # 2. Limiarização adaptativa
        # Núcleos são regiões mais escuras (mais hematoxilina)
        binary = cv2.adaptiveThreshold(smoothed, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=11,
                                     C=-2)

        # Inverte se necessário (queremos núcleos em branco)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # 3. Operações morfológicas para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Remove ruído pequeno
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_ERODE, kernel, iterations=1)
        # Fecha buracos nos núcleos
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_cross, iterations=1)

        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_cross, iterations=2)

        # 4. Remove objetos muito pequenos ou muito grandes
        # Converte para booleano para skimage
        mask_bool = cleaned.astype(bool)

        # Remove objetos pequenos (ruído)
        # mask_bool = morphology.remove_small_objects(mask_bool, min_size=14)

        # Remove objetos muito grandes (artefatos)
        labeled_temp = measure.label(mask_bool)
        regions = measure.regionprops(labeled_temp)

        for region in regions:
            if not self.is_valid_nucleus(region):
                mask_bool[labeled_temp == region.label] = False

        # 5. Watershed para separar núcleos tocantes
        # Transformada de distância
        distance = ndimage.distance_transform_edt(mask_bool)

        if distance is None or isinstance(distance, tuple):
            return None, None

        # Suaviza a transformada de distância
        distance_smooth = cv2.GaussianBlur(distance.astype(np.float32), (5, 5), 0)

        # Encontra máximos locais (centros dos núcleos)
        local_maxima_coords = peak_local_max(distance_smooth,
                                            min_distance=3,
                                            exclude_border=False,
                                            labels=mask_bool)

        # Cria máscara dos máximos
        local_maxima = np.zeros_like(distance, dtype=bool)
        local_maxima[tuple(local_maxima_coords.T)] = True

        # Marca os máximos
        markers = ndimage.label(local_maxima)[0]

        # Aplica watershed
        labeled_nuclei = watershed(-distance, markers, mask=mask_bool)

        # 6. Pós-processamento final
        # Remove núcleos na borda (geralmente incompletos)
        labeled_nuclei = self.clear_border_nuclei(labeled_nuclei)

        # Cria máscara binária final
        binary_mask = (labeled_nuclei > 0).astype(np.uint8) * 255

        num_nuclei = len(np.unique(labeled_nuclei)) - 1  # -1 para excluir fundo
        # print(f"Núcleos detectados: {num_nuclei}")

        return labeled_nuclei, binary_mask

    def clear_border_nuclei(self, labeled_image):

        h, w = labeled_image.shape
        border_labels = set()

        # Identifica labels nas bordas
        border_labels.update(np.unique(labeled_image[0, :]))    # Topo
        border_labels.update(np.unique(labeled_image[-1, :]))   # Base
        border_labels.update(np.unique(labeled_image[:, 0]))    # Esquerda
        border_labels.update(np.unique(labeled_image[:, -1]))   # Direita

        # Remove o label 0 (fundo)
        border_labels.discard(0)

        # Remove núcleos nas bordas
        cleaned = labeled_image.copy()
        for label in border_labels:
            cleaned[cleaned == label] = 0

        # Re-rotula os núcleos restantes
        cleaned_binary = cleaned > 0
        cleaned, _ = ndimage.label(cleaned_binary)

        return cleaned

    def extract_nucleus_features(self, labeled_image):
        # print("Extraindo características dos núcleos...")

        # Usa o canal de hematoxilina como imagem de intensidade
        regions = measure.regionprops(labeled_image, intensity_image=self.hematoxylin_channel)
        features_list = []

        # Primeiro, extrai centroides para cálculo de vizinhos
        centroids = [(r.centroid[0], r.centroid[1]) for r in regions]

        for i, region in enumerate(regions):
            # 1. Área
            area = region.area

            # 2. Circularidade
            perimeter = region.perimeter
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 3. Excentricidade
            eccentricity = region.eccentricity

            # 4. Distância ao núcleo mais próximo / raio
            radius = np.sqrt(area / np.pi)

            # Calcula distância ao vizinho mais próximo
            min_distance = float('inf')
            current_centroid = centroids[i]

            for j, other_centroid in enumerate(centroids):
                if i != j:
                    dist = np.sqrt((current_centroid[0] - other_centroid[0])**2 +
                                  (current_centroid[1] - other_centroid[1])**2)
                    if dist < min_distance:
                        min_distance = dist

            normalized_nn_distance = min_distance / radius if radius > 0 else 0

            # Características adicionais úteis
            features = {
                'nucleus_id': region.label,
                'area': area,
                'circularity': circularity,
                'eccentricity': eccentricity,
                'normalized_nn_distance': normalized_nn_distance,
                'perimeter': perimeter,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'solidity': region.solidity,
                'centroid_x': region.centroid[1],
                'centroid_y': region.centroid[0],
                'orientation': region.orientation
            }

            # Adiciona intensidade média apenas se disponível
            if hasattr(region, 'mean_intensity'):
                features['mean_intensity'] = region.mean_intensity

            features_list.append(features)

        # Cria DataFrame
        features_df = pd.DataFrame(features_list)

        # Adiciona densidade nuclear (núcleos por unidade de área)
        total_area = labeled_image.shape[0] * labeled_image.shape[1]
        nuclear_density = len(regions) / total_area * 10000  # por 10000 pixels
        features_df['nuclear_density'] = nuclear_density

        self.nucleus_features = features_df

        return features_df

    def calculate_statistics(self, features_df):

        if features_df.empty:
            return None

        stats_dict = {
            'num_nuclei': len(features_df),
            'area': {
                'mean': features_df['area'].mean(),
                'std': features_df['area'].std()
            },
            'circularity': {
                'mean': features_df['circularity'].mean(),
                'std': features_df['circularity'].std()
            },
            'eccentricity': {
                'mean': features_df['eccentricity'].mean(),
                'std': features_df['eccentricity'].std()
            },
            'normalized_nn_distance': {
                'mean': features_df['normalized_nn_distance'].mean(),
                'std': features_df['normalized_nn_distance'].std()
            }
        }

        return stats_dict

    def visualize_segmentation(self, labeled_image, features_df):
        """
        Visualiza os resultados da segmentação

        Args:
            labeled_image: Imagem com núcleos rotulados
            features_df: DataFrame com características
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        if self.original_image is None:
            return

        # 1. Imagem original
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Imagem H&E Original')
        axes[0, 0].axis('off')

        # 2. Canal de hematoxilina
        axes[0, 1].imshow(self.hematoxylin_channel, cmap='gray')
        axes[0, 1].set_title('Canal de Hematoxilina (Núcleos)')
        axes[0, 1].axis('off')

        # 3. Núcleos segmentados
        # Usa colormap aleatório para distinguir núcleos
        from matplotlib.colors import ListedColormap
        import matplotlib.cm as cm

        # Cria colormap aleatório
        num_labels = len(np.unique(labeled_image))
        colors = cm.rainbow(np.linspace(0, 1, num_labels))
        colors[0] = [0, 0, 0, 1]  # Fundo preto
        cmap = ListedColormap(colors)

        axes[0, 2].imshow(labeled_image, cmap=cmap)
        axes[0, 2].set_title(f'Núcleos Segmentados (n={len(features_df)})')
        axes[0, 2].axis('off')

        # 4. Overlay de contornos
        overlay = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()

        # Desenha contornos e centroides
        for _, nucleus in features_df.iterrows():
            # Cria máscara para este núcleo
            mask = (labeled_image == nucleus['nucleus_id']).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            # Desenha contorno em verde
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            # Marca centroide em vermelho
            cx, cy = int(nucleus['centroid_x']), int(nucleus['centroid_y'])
            cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Contornos e Centroides')
        axes[1, 0].axis('off')

        # 5. Scatter plot: Área vs Circularidade
        if not features_df.empty:
            scatter = axes[1, 1].scatter(features_df['area'],
                                       features_df['circularity'],
                                       c=features_df['eccentricity'],
                                       cmap='viridis',
                                       alpha=0.6)
            axes[1, 1].set_xlabel('Área (pixels)')
            axes[1, 1].set_ylabel('Circularidade')
            axes[1, 1].set_title('Características dos Núcleos')
            plt.colorbar(scatter, ax=axes[1, 1], label='Excentricidade')

        # 6. Estatísticas
        axes[1, 2].axis('off')
        stats = self.calculate_statistics(features_df)
        if stats:
            stats_text = "ESTATÍSTICAS DOS NÚCLEOS\n" + "="*35 + "\n\n"
            stats_text += f"Total de núcleos: {stats['num_nuclei']}\n\n"

            for feature in ['area', 'circularity', 'eccentricity', 'normalized_nn_distance']:
                stats_text += f"{feature.replace('_', ' ').title()}:\n"
                stats_text += f"  Média: {stats[feature]['mean']:.3f}\n"
                stats_text += f"  Desvio: {stats[feature]['std']:.3f}\n\n"

            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=11, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Análise de Segmentação de Núcleos - Imagem H&E', fontsize=16)
        plt.tight_layout()
        plt.show()

    def process_image(self, image_path, visualize=True,):
        """
        Processa completamente uma imagem H&E

        Args:
            image_path: Caminho da imagem
            visualize: Se deve mostrar visualizações
            export_path: Caminho para exportar resultados (opcional)

        Returns:
            features_df: DataFrame com características
            stats: Estatísticas resumidas
            labeled_image: Imagem com núcleos rotulados
        """
        try:
            # 1. Pré-processamento
            hematoxylin = self.preprocess_he_image(image_path)

            # 2. Segmentação
            labeled_image, binary_mask = self.segment_nuclei_he(hematoxylin)

            # 3. Extração de características
            features_df = self.extract_nucleus_features(labeled_image)

            # 4. Estatísticas
            stats = self.calculate_statistics(features_df)

            # 5. Visualização
            if visualize:
                self.visualize_segmentation(labeled_image, features_df)

            return features_df, stats, labeled_image

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            raise

class BaseALNClassifier:

    def __init__(self, patches_dir, clinical_data_path):
        self.patches_dir = Path(patches_dir)
        self.clinical_data_path = clinical_data_path
        self.clinical_data = None
        self.model = None
        self.class_weights = None
        self.num_classes = 3
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        self.image_paths = []
        
        # Será definido pelas subclasses
        self.input_size = None
        self.model_name = None
        
        # Carrega dados clínicos
        self._load_clinical_data()
        
    def _load_clinical_data(self):
        print("Carregando dados clínicos...")
        self.clinical_data = pd.read_csv(self.clinical_data_path)
        patches_certos = base_dir / "patches_certos.txt"
        # Mapeia ALN status para códigos numéricos
        aln_mapping = {
            'N0': 0,
            'N+(1-2)': 1, 
            'N+(>2)': 2
        }
        self.clinical_data['ALN_encoded'] = self.clinical_data['ALN status'].map(aln_mapping)
        
        print(f"Dados carregados: {len(self.clinical_data)} pacientes")
        print(f"Distribuição das classes: {self.clinical_data['ALN status'].value_counts().to_dict()}")
        try:
            with open(patches_certos, 'r') as f:
                for line in f:
                    # Divide a linha no ' - ' e pega a primeira parte (o caminho da imagem)
                    image_path = line.split(' - ')[0].strip()
                    self.image_paths.append(image_path)
        except FileNotFoundError:
            print(f"Erro: O arquivo '{patches_certos}' não foi encontrado.")
        except Exception as e:
            print(f"Ocorreu um erro ao ler o arquivo: {e}")

    def prepare_dataset(self, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        print(f"\nPreparando dataset FILTRADO para {self.model_name}...")
        print(f"  Usando APENAS as imagens selecionadas em patches_certos.txt")
        print(f"  Tamanho de entrada: {self.input_size}")

        if not self.image_paths:
            raise ValueError("Nenhuma imagem foi carregada de patches_certos.txt!")

        # Coleta informações apenas das imagens selecionadas
        patch_data = []
        patient_patch_counts = {}
        valid_images = 0
        invalid_images = 0

        print(f"\nProcessando {len(self.image_paths)} imagens selecionadas...")

        for image_path in self.image_paths:
            try:
                # Extrai ID do paciente do caminho da imagem
                patient_id = Path(image_path).parent.name
                
                # Verifica se o arquivo existe
                if not Path(image_path).exists():
                    print(f"Imagem não encontrada: {image_path}")
                    invalid_images += 1
                    continue
                
                # Busca classe do paciente nos dados clínicos
                patient_row = self.clinical_data[self.clinical_data['Patient ID'] == int(patient_id)]
                
                if patient_row.empty:
                    print(f"Paciente {patient_id} não encontrado nos dados clínicos")
                    invalid_images += 1
                    continue
                
                aln_class = int(patient_row['ALN_encoded'].iloc[0])
                
                # Conta patches por paciente
                if patient_id not in patient_patch_counts:
                    patient_patch_counts[patient_id] = 0
                patient_patch_counts[patient_id] += 1
                
                # Adiciona patch ao dataset
                patch_data.append({
                    'patch_path': str(image_path),
                    'patient_id': patient_id,
                    'class': aln_class,
                    'class_name': self.class_names[aln_class]
                })
                
                valid_images += 1
            
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")
                invalid_images += 1
                continue
        

        if not patch_data:
            raise ValueError("Nenhum patch válido foi encontrado!")

        print(f"\nResultados do processamento:")
        print(f"  Imagens válidas: {valid_images}")
        print(f"  Imagens inválidas: {invalid_images}")
        # Converte para DataFrame
        df_patches = pd.DataFrame(patch_data)
        print(f"\nEstatísticas dos patches selecionados:")
        print(f"  Total de patches: {len(df_patches)}")
        print(f"  Pacientes com patches: {len(patient_patch_counts)}")
        print(f"  Média patches/paciente: {np.mean(list(patient_patch_counts.values())):.1f}")
        print(f"  Máximo patches/paciente: {np.max(list(patient_patch_counts.values()))}")
        print(f"  Mínimo patches/paciente: {np.min(list(patient_patch_counts.values()))}")

        # Distribuição por classe
        print("\nDistribuição de patches por classe:")
        for class_name, count in df_patches['class_name'].value_counts().items():
            percentage = (count / len(df_patches)) * 100
            print(f"  {class_name}: {count} patches ({percentage:.1f}%)")

        # DIVISÃO POR PACIENTE (não por patch!)
        print(f"\nDividindo dataset por paciente...")

        # Lista única de pacientes e suas classes
        unique_patients = df_patches['patient_id'].unique()
        patient_classes = []

        for pid in unique_patients:
            patient_class = df_patches[df_patches['patient_id'] == pid]['class'].iloc[0]
            patient_classes.append(patient_class)

        #  stratify
        stratify_train_val = patient_classes
            
        # Primeira divisão: 80% treino+val, 20% teste
        try:
            patients_train_val, patients_test, y_train_val, y_test = train_test_split(
                unique_patients,
                patient_classes,
                test_size=test_size,
                random_state=42,
                stratify=stratify_train_val
            )
        except ValueError as e:
            print(f"Erro no stratify, usando divisão aleatória: {e}")
            patients_train_val, patients_test, y_train_val, y_test = train_test_split(
                unique_patients,
                patient_classes,
                test_size=test_size,
                random_state=42,
                stratify=None
            )

        # Verifica stratify para segunda divisão
        train_val_class_counts = pd.Series(y_train_val).value_counts()
        min_train_val_count = train_val_class_counts.min()
        
        if min_train_val_count < 2:
            stratify_train = None
        else:
            stratify_train = y_train_val

        # Segunda divisão: 75% treino, 25% validação (do conjunto treino+val)
        try:
            patients_train, patients_val, y_train, y_val = train_test_split(
                patients_train_val,
                y_train_val,
                test_size=0.25,
                random_state=43,
                stratify=stratify_train
            )
        except ValueError as e:
            print(f"Erro no stratify para validação, usando divisão aleatória: {e}")
            patients_train, patients_val, y_train, y_val = train_test_split(
                patients_train_val,
                y_train_val,
                test_size=0.25,
                random_state=43,
                stratify=None
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
        print(f"\nDivisão concluída:")
        print(f"  Pacientes treino: {len(patients_train)} ({len(train_patches)} patches)")
        print(f"  Pacientes validação: {len(patients_val)} ({len(val_patches)} patches)")
        print(f"  Pacientes teste: {len(patients_test)} ({len(test_patches)} patches)")

        # Verifica distribuição de classes
        print("\nDistribuição de classes por conjunto:")
        for split_name, split_df in [('Treino', train_patches),
                                     ('Validação', val_patches),
                                     ('Teste', test_patches)]:
            if len(split_df) > 0:
                dist = split_df['class_name'].value_counts()
                print(f"\n  {split_name}:")
                for class_name in self.class_names:
                    count = dist.get(class_name, 0)
                    percentage = (count / len(split_df)) * 100 if len(split_df) > 0 else 0
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
            else:
                print(f"\n  {split_name}: VAZIO")

        return {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'all': df_patches
        }

    def create_data_generator(self, df_patches, batch_size, shuffle=True, augment=False):

        datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            fill_mode='nearest',
            preprocessing_function=self.preprocess_input
        )
            
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
                patience=15,
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
                f'{ct}_{self.model_name.lower()}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calcula steps
        steps_per_epoch = len(train_df) // batch_size
        validation_steps = len(val_df) // batch_size
        
        print(f"\nIniciando treinamento {self.model_name}...")
        print(f"Batch size: {batch_size}")
        print(f"Steps por época: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Stage 1: Feature extraction
        print(f"\nStage 1: Feature Extraction ({epochs} épocas)")
        
        # Reconstrói modelo sem fine-tuning
        self.build_model(fine_tune=False)
        
        history_fe = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            # class_weight=self.class_weights,
            verbose=1
        )
        
        # Stage 2: Fine-tuning
        if fine_tune_epochs > 0:
            print(f"\nStage 2: Fine-tuning ({fine_tune_epochs} épocas)")
            
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
                    f'{ct}_{self.model_name.lower()}_finetuned.h5',
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
                # class_weight=self.class_weights,
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
            
        print("\nTreinamento concluído!")
        
        return history
        
    def evaluate_model(self, test_df, batch_size=None):

        print("\nAvaliando modelo no conjunto de teste...")
        
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
        
        print(f"\nAcurácia no teste: {accuracy:.4f}")
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
        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")
        
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

    def __init__(self, patches_dir, clinical_data_path):
        super().__init__(patches_dir, clinical_data_path)
        self.input_size = (256, 256)
        self.model_name = "InceptionV3"
        self.preprocess_input = preprocess_inception
        
    def build_model(self, fine_tune=True):

        print(f"Construindo modelo {self.model_name}...")
        
        # Modelo base
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
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
        return 16  # Inception é pesado, batch menor

class MobileNetV2Classifier(BaseALNClassifier):
    
    def __init__(self, patches_dir, clinical_data_path):
        super().__init__(patches_dir, clinical_data_path)
        self.input_size = (256, 256)
        self.model_name = "MobileNetV2"
        self.preprocess_input = preprocess_mobilenet
        
    def build_model(self, fine_tune=True):

        print(f"Construindo modelo {self.model_name}...")
        
        # Modelo base
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3),
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

        return 32  # MobileNet é leve, batch maior

def process_single_image(arquivo):

    segmenter = HENucleusSegmentation()
    try:
        # Processa imagem
        features, stats, labeled = segmenter.process_image(
            arquivo,
            visualize=False,  # Não visualiza em batch
        )
        return {
            'image_path': arquivo,
            'features': features,
            'statistics': stats,
            'labeled_image': labeled,
            'success': True
        }
    except Exception as e:
        print(f"Erro ao processar {arquivo}: {str(e)}")
        return {
            'image_path': arquivo,
            'error': str(e),
            'success': False
        }

def plot_figures(general_stats, rows):
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv('stats_imagens.csv', index=False)
    print(f"Salvo {len(df_stats)} linhas em stats_imagens.csv")

    colors = ['black', 'blue', 'red']

    plt.subplot(2,3, 1)
    for i in range(3):
        x = general_stats['area'][i]
        y = general_stats['circularity'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Área Média")
    plt.ylabel("Circularidade Média")

    plt.subplot(2,3, 2)
    for i in range(3):
        x = general_stats['area'][i]
        y = general_stats['eccentricity'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Área Média")
    plt.ylabel("Excentricidade Média")

    plt.subplot(2,3, 3)
    for i in range(3):
        x = general_stats['circularity'][i]
        y = general_stats['eccentricity'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Circularidade Média")
    plt.ylabel("Excentricidade Média")

    plt.subplot(2,3, 4)
    for i in range(3):
        x = general_stats['circularity'][i]
        y = general_stats['normalized_nn_distance'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Circularidade Média")
    plt.ylabel("Distância Normalizada")

    plt.subplot(2,3, 5)
    for i in range(3):
        x = general_stats['eccentricity'][i]
        y = general_stats['normalized_nn_distance'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Excentricidade Média")
    plt.ylabel("Distância Normalizada")

    plt.subplot(2,3, 6)
    for i in range(3):
        x = general_stats['area'][i]
        y = general_stats['normalized_nn_distance'][i]
        plt.scatter(x, y, color=colors[i])

    plt.xlabel("Área Média")
    plt.ylabel("Distância Normalizada")

    plt.tight_layout()
    plt.show()

def process_image_to_csv_segment(max_images=1059, segmenter=None, image_paths=[]):
    #Processa todas as imagens e cria um csv com os atributos dela
    df = pd.read_csv(clinical_data_path)
    general_stats = {"area": [[], [], []], "circularity": [[], [], []], "eccentricity": [[],[], []], "normalized_nn_distance": [[], [], []]}
    count = 0
    rows = []
    for image in image_paths:
        if count < max_images:
            patient_id = Path(image).parent.name
            patient_class = df.loc[df['Patient ID'] == int(patient_id), 'ALN status'].values[0]
            features_df, stats, labeled = segmenter.process_image(image, visualize=False)
            num_nuclei = len(features_df)
            area_mean = features_df['area'].mean()
            area_std = features_df['area'].std()
            circ_mean = features_df['circularity'].mean()
            circ_std = features_df['circularity'].std()
            ecc_mean = features_df['eccentricity'].mean()
            ecc_std = features_df['eccentricity'].std()
            nnd_mean = features_df['normalized_nn_distance'].mean()
            nnd_std = features_df['normalized_nn_distance'].std()
            rows.append({
            'patient_id': patient_id,
            'num_nuclei': num_nuclei,
            'area_mean': area_mean,
            'area_std': area_std,
            'circularity_mean': circ_mean,
            'circularity_std': circ_std,
            'eccentricity_mean': ecc_mean,
            'eccentricity_std': ecc_std,
            'normalized_nn_distance_mean': nnd_mean,
            'normalized_nn_distance_std': nnd_std
            })
            if patient_class == "N0":
                general_stats["area"][0].append(stats["area"]['mean'])
                general_stats["circularity"][0].append(stats["circularity"]['mean'])
                general_stats["normalized_nn_distance"][0].append(stats["normalized_nn_distance"]['mean'])
                general_stats["eccentricity"][0].append(stats["eccentricity"]['mean'])
            elif patient_class == "N+(>2)":
                general_stats["area"][2].append(stats["area"]['mean'])
                general_stats["circularity"][2].append(stats["circularity"]['mean'])
                general_stats["normalized_nn_distance"][2].append(stats["normalized_nn_distance"]['mean'])
                general_stats["eccentricity"][2].append(stats["eccentricity"]['mean'])
            else:
                general_stats["area"][1].append(stats["area"]['mean'])
                general_stats["circularity"][1].append(stats["circularity"]['mean'])
                general_stats["normalized_nn_distance"][1].append(stats["normalized_nn_distance"]['mean'])
                general_stats["eccentricity"][1].append(stats["eccentricity"]['mean'])
        count+=1
    plot_figures(general_stats, rows)
    # print(general_stats)
# Função auxiliar para processar múltiplas imagens
def process_batch_images(dir_path, output_dir=None, max_workers=None, max_images=100):
    # Coleta todos os arquivos de imagem primeiro
    image_files = []
    contador = 0

    for arquivo in dir_path.rglob('*'):
        if arquivo.is_file() and arquivo.suffix.lower() == '.jpg':
            image_files.append(arquivo)
            contador += 1

    print(f"Processando {len(image_files)} imagens em paralelo...")

    # Define número de workers se não especificado
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(image_files))

    all_results = []
    selected_patch_from_patient = {}

    # Processa imagens em paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submete todas as tarefas
        future_to_file = {executor.submit(process_single_image, arquivo): arquivo
                          for arquivo in image_files}

        # Coleta resultados conforme vão sendo completados
        for future in concurrent.futures.as_completed(future_to_file):
            arquivo = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)

                # Se processou com sucesso, verifica se é o melhor patch do paciente
                if result['success']:
                    patient_name = arquivo.parent.name
                    if patient_name in selected_patch_from_patient:
                        current_best = selected_patch_from_patient[patient_name]
                        if result['statistics']['num_nuclei'] > current_best['statistics']['num_nuclei']:
                            selected_patch_from_patient[patient_name] = result
                    else:
                        selected_patch_from_patient[patient_name] = result

                # print(f"Processado: {arquivo.name} - Status: {'Sucesso' if result['success'] else 'Erro'}")

            except Exception as exc:
                print(f"Erro inesperado ao processar {arquivo}: {exc}")
                all_results.append({
                    'image_path': arquivo,
                    'error': str(exc),
                    'success': False
                })

    # Salva os melhores patches por paciente
    with open("patches_certos_claude.txt", "w", encoding="utf-8") as f:
        for paciente in selected_patch_from_patient.values():
            f.write(f"{paciente['image_path']} - {paciente['statistics']['num_nuclei']}\n")

    # print([f"paciente: {x['image_path']} {x['statistics']['num_nuclei']}"
    #        for x in selected_patch_from_patient.values()])

    return all_results

def segmentation_run(max_rows=1059):
    segmenter = HENucleusSegmentation()
    image_paths = []
    try:
        with open(patches_certos, 'r') as f:
            for line in f:
                # Divide a linha no ' - ' e pega a primeira parte (o caminho da imagem)
                image_path = line.split(' - ')[0].strip()
                image_paths.append(image_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{patches_certos}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")

    process_image_to_csv_segment(max_rows, segmenter, image_paths)

def train_deep_model(model_type="inception"):

    # Verifica arquivos
    if not patches_dir.exists():
        print(f"Diretório de patches não encontrado: {patches_dir}")
        return
        
    if not clinical_data_path.exists():
        print(f"Arquivo CSV não encontrado: {clinical_data_path}")
        return
    
    # Cria classificador apropriado
    if model_type.lower() == "inception":
        classifier = InceptionV3Classifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        epochs_fe = 30  # Feature extraction
        epochs_ft = 10  # Fine-tuning
    else:  # mobilenet
        classifier = MobileNetV2Classifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        epochs_fe = 30  # Feature extraction
        epochs_ft = 20  # Fine-tuning
    
    print(f"\n{'='*80}")
    print(f"TREINAMENTO {classifier.model_name.upper()} - DATASET COMPLETO")
    print(f"{'='*80}")
    
    # Prepara dataset completo
    print("\nPREPARAÇÃO DOS DADOS")
    datasets = classifier.prepare_dataset(test_size=0.2)
    
    # Constrói e treina modelo
    # print(f"\n🏗️ CONSTRUÇÃO E TREINAMENTO DO MODELO")
    # history = classifier.train_model(
    #     datasets=datasets,
    #     epochs=epochs_fe,
    #     fine_tune_epochs=epochs_ft
    # )
    
    # # Avalia modelo
    # print("\n📈 AVALIAÇÃO FINAL")
    # results = classifier.evaluate_model(datasets['test'])
    
    # # Visualizações
    # print("\nGERANDO VISUALIZAÇÕES")
    # classifier.plot_training_history(history)
    # classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # # Salva modelo final
    # model_path = base_dir / f"{classifier.model_name.lower()}_aln_final.h5"
    # classifier.save_model(str(model_path))
    
    # # Relatório final
    # print(f"\n{'='*60}")
    # print("📋 RELATÓRIO FINAL")
    # print(f"{'='*60}")
    # print(f"Modelo: {classifier.model_name}")
    # print(f"Acurácia final: {results['accuracy']:.4f}")
    # print(f"💾 Modelo salvo em: {model_path}")
    
    # # Métricas por classe
    # report = results['classification_report']
    # print("\nMétricas por classe:")
    # for class_name in classifier.class_names:
    #     if class_name in report:
    #         metrics = report[class_name]
    #         print(f"  {class_name}: P={metrics['precision']:.3f}, "
    #               f"R={metrics['recall']:.3f}, "
    #               f"F1={metrics['f1-score']:.3f}, "
    #               f"N={metrics['support']}")
    
    # # Estatísticas do dataset
    # print(f"\n📈 Estatísticas do treinamento:")
    # print(f"  Total de patches: {len(datasets['all'])}")
    # print(f"  Patches treino: {len(datasets['train'])}")
    # print(f"  Patches validação: {len(datasets['val'])}")
    # print(f"  Patches teste: {len(datasets['test'])}")
    # print(f"  Épocas totais: {epochs_fe + epochs_ft}")

    return classifier, history, results

def main():
    print("\n" + "="*80)
    print("CLASSIFICAÇÃO DE METÁSTASE ALN - MODELOS PROFUNDOS")
    print("="*80)
    print("\nEscolha o modelo para treinar com o dataset COMPLETO:")
    print("1. Inception V3 (máxima acurácia, ~23.8M parâmetros)")
    print("2. MobileNet V2 (eficiente, ~3.5M parâmetros)")
    print("\n0. Sair")
    
    choice = input("\nDigite sua escolha (0-2): ").strip()
    
    if choice == "1":
        print("\nIniciando treinamento com INCEPTION V3...")
        train_deep_model("inception")
            
    elif choice == "2":
        print("\nIniciando treinamento com MOBILENET V2...")
        train_deep_model("mobilenet")
            
    elif choice == "0":
        print("\nSaindo...")
        return
        
    else:
        print("\nOpção inválida!")
        return
    
    print("\nProcesso concluído!")

# Exemplo de uso
if __name__ == "__main__":
    # Para uma única imagem
    segmentation_run(20)
