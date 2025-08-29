# XGBoost Integration Script
# Integra dados clínicos com features extraídos das imagens

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from segmentation.segmentation import HENucleusSegmentation
import warnings
warnings.filterwarnings('ignore')

class BreastCancerDataIntegration:
    """
    Classe para integrar dados clínicos com features morfológicos extraídos das imagens
    """
    
    def __init__(self, csv_path, patches_base_path):
        """
        Inicializa a integração de dados
        
        Args:
            csv_path: Caminho do arquivo CSV com dados clínicos
            patches_base_path: Caminho base das pastas de patches
        """
        self.csv_path = csv_path
        self.patches_base_path = patches_base_path
        self.clinical_data = None
        self.image_features = []
        self.final_dataset = None
        
        # Carrega dados clínicos
        self.load_clinical_data()
        
    def load_clinical_data(self):
        """
        Carrega e processa dados clínicos do CSV
        """
        print("Carregando dados clínicos...")
        
        # Lê o CSV
        self.clinical_data = pd.read_csv(self.csv_path)
        
        # Limpa e processa colunas
        self.clinical_data['Patient ID'] = self.clinical_data['Patient ID'].astype(str)
        
        # Converte ALN status para encoding numérico
        aln_mapping = {
            'N0': 0,        # Sem metástase
            'N+(1-2)': 1,   # 1-2 linfonodos positivos
            'N+(>2)': 2     # Mais de 2 linfonodos positivos
        }
        self.clinical_data['ALN_status_encoded'] = self.clinical_data['ALN status'].map(aln_mapping)
        
        # Processa tamanho do tumor (remove vírgulas)
        self.clinical_data['Tumour Size(cm)'] = self.clinical_data['Tumour Size(cm)'].astype(str)
        self.clinical_data['Tumour Size(cm)'] = self.clinical_data['Tumour Size(cm)'].str.replace(',', '.')
        self.clinical_data['Tumor_Size_Numeric'] = pd.to_numeric(
            self.clinical_data['Tumour Size(cm)'], errors='coerce'
        )
        
        # Encoding categórico
        categorical_columns = ['ER', 'PR', 'HER2', 'Tumour Type', 'Molecular subtype']
        for col in categorical_columns:
            self.clinical_data[f'{col}_encoded'] = pd.Categorical(
                self.clinical_data[col]
            ).codes
            
        print(f"Dados clínicos carregados: {len(self.clinical_data)} pacientes")
        print(f"Distribuição ALN status: {self.clinical_data['ALN status'].value_counts().to_dict()}")
        
    def get_patient_patches(self, patient_id):
        """
        Obtém lista de patches para um paciente específico
        
        Args:
            patient_id: ID do paciente
            
        Returns:
            list: Lista de caminhos dos patches
        """
        patient_folder = os.path.join(self.patches_base_path, str(patient_id))
        
        if not os.path.exists(patient_folder):
            return []
            
        # Busca por arquivos .jpg na pasta do paciente
        patch_files = glob.glob(os.path.join(patient_folder, "*.jpg"))
        return sorted(patch_files)
    
    def extract_features_from_patches(self, patient_id, max_patches=None):
        """
        Extrai features de todos os patches de um paciente
        
        Args:
            patient_id: ID do paciente
            max_patches: Número máximo de patches a processar (None = todos)
            
        Returns:
            dict: Features agregados do paciente
        """
        patches = self.get_patient_patches(patient_id)
        
        if not patches:
            print(f"Nenhum patch encontrado para paciente {patient_id}")
            return self._get_empty_features(patient_id)
        
        # Limita número de patches se especificado
        if max_patches:
            patches = patches[:max_patches]
            
        print(f"Processando {len(patches)} patches do paciente {patient_id}")
        
        # Inicializa segmentador
        segmenter = HENucleusSegmentation()
        
        # Acumula features de todos os patches
        all_features = []
        successful_patches = 0
        
        for patch_path in patches:
            try:
                # Processa patch individual
                features_df, _, _ = segmenter.process_image(
                    patch_path, 
                    visualize=False
                )
                
                if features_df is not None and not features_df.empty:
                    all_features.append(features_df)
                    successful_patches += 1
                    
            except Exception as e:
                print(f"Erro ao processar {patch_path}: {str(e)}")
                continue
        
        if not all_features:
            print(f"Nenhum feature extraído para paciente {patient_id}")
            return self._get_empty_features(patient_id)
        
        # Concatena todos os features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Agrega features
        aggregated = self.aggregate_features_for_xgboost(combined_features, patient_id)
        aggregated['successful_patches'] = successful_patches
        
        print(f"Paciente {patient_id}: {len(combined_features)} núcleos de {successful_patches} patches")
        
        return aggregated
    
    def aggregate_features_for_xgboost(self, features_df, patient_id):
        """
        Agrega características dos núcleos para uso no XGBoost
        Baseado na função que você forneceu, com melhorias
        
        Args:
            features_df: DataFrame com features dos núcleos
            patient_id: ID do paciente
            
        Returns:
            dict: Features agregados
        """
        if features_df.empty:
            return self._get_empty_features(patient_id)
        
        # Características principais do projeto
        aggregated_features = {
            'patient_id': patient_id,
            'num_nuclei': len(features_df),
            
            # As 4 características principais especificadas
            'area_mean': features_df['area'].mean(),
            'area_std': features_df['area'].std(),
            'circularity_mean': features_df['circularity'].mean(), 
            'circularity_std': features_df['circularity'].std(),
            'eccentricity_mean': features_df['eccentricity'].mean(),
            'eccentricity_std': features_df['eccentricity'].std(),
            'normalized_nn_distance_mean': features_df['normalized_nn_distance'].mean(),
            'normalized_nn_distance_std': features_df['normalized_nn_distance'].std(),
            
            # Características extras úteis para classificação
            'nuclear_density': len(features_df) / (256 * 256) * 10000,  # Normalizado por patch
            'solidity_mean': features_df['solidity'].mean() if 'solidity' in features_df.columns else 0,
            'solidity_std': features_df['solidity'].std() if 'solidity' in features_df.columns else 0,
            'orientation_std': features_df['orientation'].std() if 'orientation' in features_df.columns else 0,
            
            # Percentis para capturar distribuição
            'area_p25': features_df['area'].quantile(0.25),
            'area_p75': features_df['area'].quantile(0.75),
            'circularity_p25': features_df['circularity'].quantile(0.25),
            'circularity_p75': features_df['circularity'].quantile(0.75),
            
            # Razões e proporções
            'large_nuclei_ratio': (features_df['area'] > features_df['area'].quantile(0.8)).mean(),
            'irregular_nuclei_ratio': (features_df['circularity'] < 0.7).mean(),
            'eccentric_nuclei_ratio': (features_df['eccentricity'] > 0.8).mean(),
        }
        
        # Substitui NaN por 0
        for key, value in aggregated_features.items():
            if pd.isna(value):
                aggregated_features[key] = 0.0
                
        return aggregated_features
    
    def _get_empty_features(self, patient_id):
        """
        Retorna features vazios para pacientes sem dados de imagem
        """
        return {
            'patient_id': patient_id,
            'num_nuclei': 0,
            'area_mean': 0.0,
            'area_std': 0.0,
            'circularity_mean': 0.0,
            'circularity_std': 0.0,
            'eccentricity_mean': 0.0,
            'eccentricity_std': 0.0,
            'normalized_nn_distance_mean': 0.0,
            'normalized_nn_distance_std': 0.0,
            'nuclear_density': 0.0,
            'solidity_mean': 0.0,
            'solidity_std': 0.0,
            'orientation_std': 0.0,
            'area_p25': 0.0,
            'area_p75': 0.0,
            'circularity_p25': 0.0,
            'circularity_p75': 0.0,
            'large_nuclei_ratio': 0.0,
            'irregular_nuclei_ratio': 0.0,
            'eccentric_nuclei_ratio': 0.0,
            'successful_patches': 0
        }
    
    def process_all_patients(self, max_patients=None, max_patches_per_patient=5):
        """
        Processa todos os pacientes extraindo features das imagens
        
        Args:
            max_patients: Número máximo de pacientes (None = todos)
            max_patches_per_patient: Máximo de patches por paciente
        """
        print("\nIniciando processamento de todos os pacientes...")
        
        patient_ids = self.clinical_data['Patient ID'].unique()
        
        if max_patients:
            patient_ids = patient_ids[:max_patients]
            
        print(f"Processando {len(patient_ids)} pacientes...")
        
        for i, patient_id in enumerate(patient_ids):
            print(f"\n[{i+1}/{len(patient_ids)}] Processando paciente {patient_id}")
            
            try:
                features = self.extract_features_from_patches(
                    patient_id, 
                    max_patches=max_patches_per_patient
                )
                self.image_features.append(features)
                
            except Exception as e:
                print(f"Erro ao processar paciente {patient_id}: {str(e)}")
                # Adiciona features vazios para manter consistência
                self.image_features.append(self._get_empty_features(patient_id))
                
        print(f"\nProcessamento concluído: {len(self.image_features)} pacientes processados")
    
    def integrate_data(self):
        """
        Integra dados clínicos com features das imagens
        """
        print("\nIntegrando dados clínicos com features das imagens...")
        
        # Converte features das imagens para DataFrame
        image_features_df = pd.DataFrame(self.image_features)
        image_features_df['Patient ID'] = image_features_df['patient_id'].astype(str)
        
        # Merge com dados clínicos
        self.final_dataset = self.clinical_data.merge(
            image_features_df, 
            on='Patient ID', 
            how='left'
        )
        
        # Preenche valores faltantes com 0 para pacientes sem imagens
        feature_columns = [col for col in image_features_df.columns 
                          if col not in ['patient_id', 'Patient ID']]
        
        for col in feature_columns:
            if col in self.final_dataset.columns:
                self.final_dataset[col] = self.final_dataset[col].fillna(0.0)
        
        print(f"Dataset final: {len(self.final_dataset)} registros")
        print(f"Features de imagem: {len(feature_columns)} colunas")
        
        return self.final_dataset
    
    def prepare_xgboost_data(self):
        """
        Prepara dados finais para treinamento do XGBoost
        """
        if self.final_dataset is None:
            raise ValueError("Execute integrate_data() primeiro")
        
        print("\nPreparando dados para XGBoost...")
        
        # Seleciona features para o modelo
        feature_columns = [
            # Dados clínicos
            'Age(years)',
            'Tumor_Size_Numeric',
            'ER_encoded',
            'PR_encoded', 
            'HER2_encoded',
            'Tumour Type_encoded',
            'Molecular subtype_encoded',
            
            # Features das imagens (morfológicos)
            'num_nuclei',
            'area_mean',
            'area_std',
            'circularity_mean',
            'circularity_std',
            'eccentricity_mean',
            'eccentricity_std',
            'normalized_nn_distance_mean',
            'normalized_nn_distance_std',
            'nuclear_density',
            'solidity_mean',
            'solidity_std',
            'orientation_std',
            'area_p25',
            'area_p75',
            'circularity_p25',
            'circularity_p75',
            'large_nuclei_ratio',
            'irregular_nuclei_ratio',
            'eccentric_nuclei_ratio'
        ]
        
        # Verifica quais colunas existem
        available_features = [col for col in feature_columns 
                             if col in self.final_dataset.columns]
        
        missing_features = [col for col in feature_columns 
                           if col not in self.final_dataset.columns]
        
        if missing_features:
            print(f"Features faltantes: {missing_features}")
            
        # Prepara X e y
        X = self.final_dataset[available_features].copy()
        y = self.final_dataset['ALN_status_encoded'].copy()
        
        # Remove linhas com target NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Preenche NaN nas features com 0
        X = X.fillna(0.0)
        
        print(f"Dataset final: {len(X)} amostras, {len(available_features)} features")
        print(f"Distribuição das classes: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def export_dataset(self, output_path="integrated_dataset.csv"):
        """
        Exporta dataset integrado para CSV
        """
        if self.final_dataset is None:
            raise ValueError("Execute integrate_data() primeiro")
            
        self.final_dataset.to_csv(output_path, index=False)
        print(f"Dataset exportado para: {output_path}")
        
        # Salva também um resumo
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("RESUMO DO DATASET INTEGRADO\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total de pacientes: {len(self.final_dataset)}\n")
            f.write(f"Distribuição ALN status:\n{self.final_dataset['ALN status'].value_counts()}\n\n")
            
            # Features das imagens
            image_features = [col for col in self.final_dataset.columns 
                             if any(feat in col for feat in ['area_', 'circularity_', 'eccentricity_', 'normalized_nn_'])]
            f.write(f"Features morfológicos extraídos: {len(image_features)}\n")
            for feat in image_features:
                f.write(f"  - {feat}\n")
                
        print(f"Resumo salvo em: {summary_path}")


# Função principal para execução
def main():
    """
    Função principal para executar o pipeline completo
    """
    # Configuração dos caminhos
    csv_path = "/home/leonardo/Documents/PUC/6. Semestre VI/Processamento e Análise de Imagens/PAI/patient-clinical-data.csv"
    patches_path = "/home/leonardo/Documents/PUC/6. Semestre VI/Processamento e Análise de Imagens/PAI/paper_patches/patches"
    
    # Inicializa integração
    integrator = BreastCancerDataIntegration(csv_path, patches_path)
    
    # Processa pacientes (limitado para teste)
    print("Iniciando processamento de amostras...")
    integrator.process_all_patients(
        max_patients=10,  # Para teste inicial
        max_patches_per_patient=3
    )
    
    # Integra dados
    final_dataset = integrator.integrate_data()
    
    # Prepara para XGBoost
    X, y, features = integrator.prepare_xgboost_data()
    
    # Exporta resultados
    integrator.export_dataset("breast_cancer_integrated.csv")
    
    return integrator, X, y, features


if __name__ == "__main__":
    integrator, X, y, features = main()
