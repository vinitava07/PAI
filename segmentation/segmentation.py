import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
import pandas as pd
from pathlib import Path
import concurrent.futures
import multiprocessing

class HENucleusSegmentation:
    """
    Segmentação especializada para imagens histológicas coradas com H&E
    Otimizada para detectar núcleos (estruturas roxas/azuis)
    """

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
        """
        Pré-processa imagem H&E separando os canais de cor

        Args:
            image_path: Caminho da imagem H&E

        Returns:
            hematoxylin_channel: Canal de hematoxilina isolado
        """
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
        """
        Segmenta núcleos do canal de hematoxilina

        Args:
            hematoxylin_channel: Canal isolado de hematoxilina

        Returns:
            labeled_nuclei: Núcleos rotulados
            binary_mask: Máscara binária dos núcleos
        """
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
        """
        Remove núcleos que tocam as bordas da imagem

        Args:
            labeled_image: Imagem com núcleos rotulados

        Returns:
            cleaned_image: Imagem sem núcleos nas bordas
        """
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
        """
        Extrai características dos núcleos conforme especificado no projeto

        Args:
            labeled_image: Imagem com núcleos rotulados

        Returns:
            features_df: DataFrame com características de cada núcleo
        """
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
        """
        Calcula estatísticas resumidas das características

        Args:
            features_df: DataFrame com características dos núcleos

        Returns:
            stats_dict: Dicionário com média e desvio padrão
        """
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


def process_single_image(arquivo):
    """
    Função auxiliar para processar uma única imagem
    Usada para paralelização

    Args:
        arquivo: Path object do arquivo de imagem

    Returns:
        dict: Resultado do processamento ou erro
    """
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


# Função auxiliar para processar múltiplas imagens
def process_batch_images(dir_path, output_dir=None, max_workers=None, max_images=100):
    """
    Processa múltiplas imagens H&E em paralelo

    Args:
        dir_path: Diretório com as imagens
        output_dir: Diretório para salvar resultados
        max_workers: Número máximo de processos paralelos (None = auto)
        max_images: Número máximo de imagens para processar

    Returns:
        all_results: Lista com resultados de cada imagem
    """
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


# Exemplo de uso
if __name__ == "__main__":
    # Para uma única imagem
    segmenter = HENucleusSegmentation()

    # Substitua pelo caminho da sua imagem
    base_dir = Path(__file__).parent.parent
    patches_dir = base_dir / "patches"
    clinical_data_path = base_dir / "patient-clinical-data.csv"

    # Processa a imagem
    # features, stats, labeled = segmenter.process_image(
    #     image_path,
    #     visualize=True,
    # )

    all_results = process_batch_images(
        patches_dir,
        max_workers=None  # ou especifique um número como 4, 8, etc.
    )    # print(all_results)
    print("cabo")

    # Exibe resumo
    # if stats:
    #     print("\n" + "="*50)
    #     print("RESUMO DA ANÁLISE")
    #     print("="*50)
    #     print(f"Total de núcleos detectados: {stats['num_nuclei']}")
    #     print(f"Área média: {stats['area']['mean']:.2f} ± {stats['area']['std']:.2f} pixels")
    #     print(f"Circularidade média: {stats['circularity']['mean']:.3f} ± {stats['circularity']['std']:.3f}")
    #     print(f"Excentricidade média: {stats['eccentricity']['mean']:.3f} ± {stats['eccentricity']['std']:.3f}")
    #     print(f"Distância NN normalizada: {stats['normalized_nn_distance']['mean']:.3f} ± {stats['normalized_nn_distance']['std']:.3f}")
