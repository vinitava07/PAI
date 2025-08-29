#!/usr/bin/env python3
"""
Script para fazer predições usando um modelo Inception V3 treinado
Permite prever ALN status para novos pacientes
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional


class InceptionV3Predictor:
    """
    Classe para fazer predições com modelo Inception V3 treinado
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o preditor
        
        Args:
            model_path: Caminho para o modelo .h5 salvo
        """
        self.model_path = Path(model_path)
        self.model = None
        self.input_size = (299, 299)
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        
        # Carrega modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo salvo"""
        print(f"📁 Carregando modelo: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        self.model = load_model(str(self.model_path))
        print("✅ Modelo carregado com sucesso!")
        
        # Exibe informações do modelo
        print(f"   - Input shape: {self.model.input_shape}")
        print(f"   - Output shape: {self.model.output_shape}")
        print(f"   - Total parâmetros: {self.model.count_params():,}")
    
    def predict_patch(self, image_path: str, show_image: bool = False) -> Dict:
        """
        Faz predição para um único patch
        
        Args:
            image_path: Caminho da imagem
            show_image: Se deve mostrar a imagem
            
        Returns:
            Dicionário com predições
        """
        # Carrega e pré-processa imagem
        img = load_img(image_path, target_size=self.input_size)
        
        if show_image:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Patch: {Path(image_path).name}")
            plt.axis('off')
            plt.show()
        
        # Converte para array
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Faz predição
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: float(predictions[i]) 
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_patient(self, patient_patches: List[str], 
                       aggregation: str = 'mean',
                       show_results: bool = True) -> Dict:
        """
        Faz predição para um paciente baseado em múltiplos patches
        
        Args:
            patient_patches: Lista de caminhos dos patches
            aggregation: Método de agregação ('mean', 'max', 'vote')
            show_results: Se deve mostrar resultados
            
        Returns:
            Dicionário com predição final
        """
        print(f"\n🔮 Predizendo para paciente com {len(patient_patches)} patches...")
        
        # Predições individuais
        patch_predictions = []
        patch_probs = []
        
        for patch_path in patient_patches:
            # Carrega e processa
            img = load_img(patch_path, target_size=self.input_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predição
            pred = self.model.predict(img_array, verbose=0)[0]
            patch_predictions.append(pred)
            patch_probs.append(pred)
        
        patch_predictions = np.array(patch_predictions)
        
        # Agregação
        if aggregation == 'mean':
            final_probs = np.mean(patch_predictions, axis=0)
        elif aggregation == 'max':
            final_probs = np.max(patch_predictions, axis=0)
        elif aggregation == 'vote':
            # Voto majoritário
            patch_classes = np.argmax(patch_predictions, axis=1)
            final_class = np.bincount(patch_classes).argmax()
            final_probs = np.zeros(len(self.class_names))
            final_probs[final_class] = 1.0
        else:
            raise ValueError(f"Método de agregação inválido: {aggregation}")
        
        # Resultado final
        final_class = np.argmax(final_probs)
        confidence = final_probs[final_class]
        
        result = {
            'num_patches': len(patient_patches),
            'aggregation_method': aggregation,
            'predicted_class': self.class_names[final_class],
            'confidence': confidence,
            'class_probabilities': {
                self.class_names[i]: float(final_probs[i]) 
                for i in range(len(self.class_names))
            },
            'patch_predictions': patch_predictions,
            'patch_votes': {
                self.class_names[i]: int(np.sum(np.argmax(patch_predictions, axis=1) == i))
                for i in range(len(self.class_names))
            }
        }
        
        if show_results:
            self._visualize_patient_results(result)
        
        return result
    
    def predict_directory(self, directory_path: str, 
                         aggregation: str = 'mean') -> Dict:
        """
        Faz predição para todos os patches em um diretório
        
        Args:
            directory_path: Caminho do diretório
            aggregation: Método de agregação
            
        Returns:
            Dicionário com predição
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Diretório não encontrado: {directory}")
        
        # Lista patches
        patches = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        
        if not patches:
            raise ValueError(f"Nenhum patch encontrado em: {directory}")
        
        print(f"📁 Encontrados {len(patches)} patches em: {directory.name}")
        
        # Faz predição
        return self.predict_patient(
            [str(p) for p in patches], 
            aggregation=aggregation
        )
    
    def batch_predict(self, csv_path: str, patches_dir: str,
                     output_path: str = "predictions.csv") -> pd.DataFrame:
        """
        Faz predições para múltiplos pacientes listados em CSV
        
        Args:
            csv_path: Caminho do CSV com Patient ID
            patches_dir: Diretório base dos patches
            output_path: Onde salvar as predições
            
        Returns:
            DataFrame com predições
        """
        print(f"\n📋 Processando pacientes do CSV: {csv_path}")
        
        # Lê CSV
        df = pd.read_csv(csv_path)
        patches_dir = Path(patches_dir)
        
        results = []
        
        for _, row in df.iterrows():
            patient_id = str(row['Patient ID'])
            patient_dir = patches_dir / patient_id
            
            if not patient_dir.exists():
                print(f"⚠️  Diretório não encontrado para paciente {patient_id}")
                continue
            
            try:
                # Predição
                result = self.predict_directory(str(patient_dir))
                
                # Adiciona ao resultado
                results.append({
                    'Patient ID': patient_id,
                    'Predicted_ALN': result['predicted_class'],
                    'Confidence': result['confidence'],
                    'Prob_N0': result['class_probabilities']['N0'],
                    'Prob_N1_2': result['class_probabilities']['N+(1-2)'],
                    'Prob_N_gt2': result['class_probabilities']['N+(>2)'],
                    'Num_Patches': result['num_patches']
                })
                
            except Exception as e:
                print(f"❌ Erro ao processar paciente {patient_id}: {e}")
        
        # Cria DataFrame
        results_df = pd.DataFrame(results)
        
        # Salva resultados
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Predições salvas em: {output_path}")
        
        return results_df
    
    def _visualize_patient_results(self, result: Dict):
        """Visualiza resultados de predição para um paciente"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de probabilidades
        classes = list(result['class_probabilities'].keys())
        probs = list(result['class_probabilities'].values())
        
        bars = ax1.bar(classes, probs)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Probabilidade')
        ax1.set_title('Probabilidades por Classe')
        
        # Colore a barra da classe predita
        predicted_idx = classes.index(result['predicted_class'])
        bars[predicted_idx].set_color('green')
        
        # Adiciona valores nas barras
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}',
                    ha='center', va='bottom')
        
        # Distribuição de votos dos patches
        votes = list(result['patch_votes'].values())
        ax2.pie(votes, labels=classes, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Distribuição de Votos dos Patches')
        
        plt.suptitle(f"Predição: {result['predicted_class']} "
                    f"(Confiança: {result['confidence']:.3f})")
        plt.tight_layout()
        plt.show()


def demo_prediction():
    """Demonstração de uso do preditor"""
    print("\n🎯 DEMONSTRAÇÃO DE PREDIÇÃO")
    print("="*60)
    
    # Procura modelo salvo
    model_files = list(Path('.').glob('models/*/best_model_final.h5'))
    
    if not model_files:
        print("❌ Nenhum modelo treinado encontrado!")
        print("   Execute primeiro: python run_inception_training.py")
        return
    
    # Usa o modelo mais recente
    model_path = sorted(model_files)[-1]
    print(f"Usando modelo: {model_path}")
    
    # Inicializa preditor
    predictor = InceptionV3Predictor(str(model_path))
    
    # Exemplo 1: Predição de um patch único
    print("\n" + "="*60)
    print("📌 Exemplo 1: Predição de patch único")
    print("="*60)
    
    # Encontra um patch de exemplo
    patches_dir = Path('patches')
    example_patch = None
    
    for patient_dir in patches_dir.iterdir():
        if patient_dir.is_dir():
            patches = list(patient_dir.glob("*.jpg"))
            if patches:
                example_patch = patches[0]
                break
    
    if example_patch:
        result = predictor.predict_patch(str(example_patch), show_image=True)
        print(f"\nResultado:")
        print(f"  Classe predita: {result['predicted_class']}")
        print(f"  Confiança: {result['confidence']:.3f}")
        print(f"  Probabilidades:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.3f}")
    
    # Exemplo 2: Predição para um paciente
    print("\n" + "="*60)
    print("📌 Exemplo 2: Predição para paciente completo")
    print("="*60)
    
    # Usa o mesmo paciente do exemplo anterior
    if example_patch:
        patient_dir = example_patch.parent
        result = predictor.predict_directory(str(patient_dir))
        
        print(f"\nPaciente: {patient_dir.name}")
        print(f"Número de patches: {result['num_patches']}")
        print(f"Classe predita: {result['predicted_class']}")
        print(f"Confiança: {result['confidence']:.3f}")


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Faz predições usando modelo Inception V3 treinado'
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Caminho para o modelo .h5'
    )
    
    parser.add_argument(
        '--patch',
        type=str,
        help='Predizer um único patch'
    )
    
    parser.add_argument(
        '--patient',
        type=str,
        help='Predizer todos os patches de um diretório/paciente'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Predizer múltiplos pacientes de um CSV'
    )
    
    parser.add_argument(
        '--patches-dir',
        type=str,
        default='patches',
        help='Diretório base dos patches (para --csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Arquivo de saída para predições em batch'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max', 'vote'],
        help='Método de agregação para múltiplos patches'
    )
    
    args = parser.parse_args()
    
    # Inicializa preditor
    predictor = InceptionV3Predictor(args.model_path)
    
    # Executa predição baseada nos argumentos
    if args.patch:
        # Predição de patch único
        result = predictor.predict_patch(args.patch, show_image=True)
        print("\nResultado:")
        print(f"  Classe: {result['predicted_class']}")
        print(f"  Confiança: {result['confidence']:.3f}")
        
    elif args.patient:
        # Predição de paciente
        result = predictor.predict_directory(
            args.patient, 
            aggregation=args.aggregation
        )
        
    elif args.csv:
        # Predição em batch
        results_df = predictor.batch_predict(
            args.csv,
            args.patches_dir,
            args.output
        )
        print(f"\nResumo das predições:")
        print(results_df['Predicted_ALN'].value_counts())
        
    else:
        print("❌ Especifique --patch, --patient ou --csv")
        parser.print_help()


if __name__ == "__main__":
    # Se executado sem argumentos, roda demonstração
    if len(sys.argv) == 1:
        demo_prediction()
    else:
        main()
