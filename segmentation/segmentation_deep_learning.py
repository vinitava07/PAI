import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class UNetNucleusSegmentation(nn.Module):
    """
    Arquitetura U-Net adaptada para segmentação de núcleos
    Baseada na abordagem do artigo para segmentação weakly supervised
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(UNetNucleusSegmentation, self).__init__()

        # Encoder (Contração)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Expansão)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Camada de saída
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        # MaxPool para downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """
        Bloco de convolução dupla com BatchNorm e ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder com skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Saída
        out = self.out(dec1)
        return torch.sigmoid(out)


class DeepLearningNucleusSegmenter:
    """
    Segmentador de núcleos usando Deep Learning
    Implementa a abordagem mencionada no artigo
    """

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetNucleusSegmentation().to(self.device)

        if model_path:
            self.load_model(model_path)

        # Transformações para pré-processamento
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Carrega modelo pré-treinado"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess_patch(self, image_patch):
        """
        Pré-processa um patch de imagem para entrada no modelo

        Args:
            image_patch: Patch de imagem (numpy array ou PIL Image)

        Returns:
            tensor: Tensor pronto para o modelo
        """
        if isinstance(image_patch, np.ndarray):
            # Converte para PIL Image se necessário
            image_patch = Image.fromarray(image_patch)

        # Redimensiona para tamanho padrão (256x256)
        image_patch = image_patch.resize((256, 256))

        # Aplica transformações
        tensor = self.transform(image_patch)

        # Adiciona dimensão do batch
        return tensor.unsqueeze(0).to(self.device)

    def segment_nuclei_dl(self, image):
        """
        Segmenta núcleos usando o modelo deep learning

        Args:
            image: Imagem completa ou patch

        Returns:
            segmentation_mask: Máscara de segmentação
        """
        # Divide a imagem em patches se for muito grande
        patches, positions = self.extract_patches(image, patch_size=256, overlap=64)

        # Processa cada patch
        segmented_patches = []

        with torch.no_grad():
            for patch in patches:
                # Pré-processa o patch
                input_tensor = self.preprocess_patch(patch)

                # Passa pelo modelo
                output = self.model(input_tensor)

                # Converte saída para numpy
                seg_patch = output.squeeze().cpu().numpy()

                # Aplica threshold
                seg_patch = (seg_patch > 0.5).astype(np.uint8)

                segmented_patches.append(seg_patch)

        # Reconstrói a imagem completa
        full_segmentation = self.reconstruct_from_patches(
            segmented_patches, positions, image.shape[:2]
        )

        return full_segmentation

    def extract_patches(self, image, patch_size=256, overlap=64):
        """
        Extrai patches da imagem com sobreposição

        Args:
            image: Imagem completa
            patch_size: Tamanho do patch
            overlap: Sobreposição entre patches

        Returns:
            patches: Lista de patches
            positions: Posições dos patches
        """
        patches = []
        positions = []

        h, w = image.shape[:2]
        stride = patch_size - overlap

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Extrai patch
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))

        # Adiciona patches das bordas se necessário
        if h % stride != 0:
            for x in range(0, w - patch_size + 1, stride):
                y = h - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))

        if w % stride != 0:
            for y in range(0, h - patch_size + 1, stride):
                x = w - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((y, x))

        return patches, positions

    def reconstruct_from_patches(self, patches, positions, original_shape):
        """
        Reconstrói a imagem completa a partir dos patches

        Args:
            patches: Lista de patches segmentados
            positions: Posições dos patches
            original_shape: Forma da imagem original

        Returns:
            full_image: Imagem reconstruída
        """
        h, w = original_shape
        full_image = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        patch_size = patches[0].shape[0]

        # Cria uma janela gaussiana para suavizar as bordas
        gaussian_window = self._create_gaussian_window(patch_size)

        for patch, (y, x) in zip(patches, positions):
            # Aplica janela gaussiana
            weighted_patch = patch * gaussian_window

            # Adiciona ao mapa completo
            full_image[y:y+patch_size, x:x+patch_size] += weighted_patch
            weight_map[y:y+patch_size, x:x+patch_size] += gaussian_window

        # Normaliza pelo peso
        full_image = np.divide(full_image, weight_map,
                              out=np.zeros_like(full_image),
                              where=weight_map != 0)

        return (full_image > 0.5).astype(np.uint8)

    def _create_gaussian_window(self, size):
        """Cria uma janela gaussiana para suavização"""
        sigma = size / 4
        ax = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.max()

    def post_process_segmentation(self, segmentation_mask):
        """
        Pós-processamento da máscara de segmentação

        Args:
            segmentation_mask: Máscara binária

        Returns:
            processed_mask: Máscara processada
            labeled_nuclei: Núcleos rotulados
        """
        # Remove pequenos objetos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)

        # Preenche buracos
        filled = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Separa núcleos tocantes usando watershed
        dist_transform = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

        # Encontra máximos locais
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Encontra componentes conectados
        _, markers = cv2.connectedComponents(sure_fg)

        # Aplica watershed
        markers = markers + 1
        markers[filled == 0] = 0

        # Converte para formato de 3 canais para watershed
        img_for_watershed = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_for_watershed, markers)

        # Cria máscara final
        processed_mask = np.zeros_like(segmentation_mask)
        processed_mask[markers > 1] = 255

        return processed_mask, markers


class NucleusFeatureExtractor:
    """
    Extrator de características dos núcleos baseado no artigo
    """

    @staticmethod
    def extract_morphometric_features(labeled_image):
        """
        Extrai características morfométricas mencionadas no artigo

        Args:
            labeled_image: Imagem com núcleos rotulados

        Returns:
            features: DataFrame com características
        """
        from skimage import measure
        import pandas as pd

        # Propriedades das regiões
        regions = measure.regionprops(labeled_image)

        features_list = []

        for region in regions:
            # Características mencionadas no artigo
            features = {
                'label': region.label,
                'area': region.area,
                'perimeter': region.perimeter,
                'major_axis': region.major_axis_length,
                'minor_axis': region.minor_axis_length,
                'orientation': region.orientation,
                'circumference': region.perimeter,  # Sinônimo de perímetro
                'circularity': 4 * np.pi * region.area / (region.perimeter ** 2),
                'rectangularity': region.area / (region.bbox[2] * region.bbox[3]),
                'centroid_y': region.centroid[0],
                'centroid_x': region.centroid[1]
            }

            # Densidade nuclear (número de núcleos por área)
            # Será calculada depois para toda a imagem

            features_list.append(features)

        df = pd.DataFrame(features_list)

        # Calcula densidade nuclear
        total_area = labeled_image.shape[0] * labeled_image.shape[1]
        nuclear_density = len(regions) / total_area * 10000  # Por 10000 pixels

        df['nuclear_density'] = nuclear_density

        return df


# Exemplo de uso integrado
def process_image_with_dl(image_path, model_path=None):
    """
    Processa uma imagem usando a abordagem de deep learning

    Args:
        image_path: Caminho da imagem
        model_path: Caminho do modelo treinado (opcional)

    Returns:
        features: Características extraídas
        segmentation: Máscara de segmentação
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicializa o segmentador
    segmenter = DeepLearningNucleusSegmenter(model_path)

    # Segmenta núcleos
    segmentation_mask = segmenter.segment_nuclei_dl(image_rgb)

    # Pós-processamento
    processed_mask, labeled_nuclei = segmenter.post_process_segmentation(segmentation_mask)

    # Extrai características
    extractor = NucleusFeatureExtractor()
    features_df = extractor.extract_morphometric_features(labeled_nuclei)

    # Visualiza resultados
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    axes[1].imshow(segmentation_mask, cmap='gray')
    axes[1].set_title('Segmentação DL')
    axes[1].axis('off')

    axes[2].imshow(labeled_nuclei, cmap='jet')
    axes[2].set_title('Núcleos Rotulados')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return features_df, processed_mask, labeled_nuclei


if __name__ == "__main__":
    # Exemplo de uso
    image_path = "path/to/biopsy_image.png"

    # Se você tiver um modelo treinado
    # model_path = "path/to/trained_model.pth"
    # features, mask, labeled = process_image_with_dl(image_path, model_path)

    # Sem modelo treinado (apenas demonstração da arquitetura)
    features, mask, labeled = process_image_with_dl(image_path)

    print(f"Total de núcleos detectados: {len(features)}")
    print("\nEstatísticas das características:")
    print(features.describe())
