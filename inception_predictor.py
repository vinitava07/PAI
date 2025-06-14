import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class InceptionALNPredictor:
    """
    Classe para predição de novos patches usando modelo Inception V4 treinado
    """
    
    def __init__(self, model_path):
        """
        Inicializa preditor
        
        Args:
            model_path: Caminho do modelo treinado
        """
        self.model_path = model_path
        self.model = None
        self.input_size = (299, 299)
        self.class_names = ['N0', 'N+(1-2)', 'N+(>2)']
        
        # Carrega modelo
        self.load_model()
        
    def load_model(self):
        """Carrega modelo treinado"""
        try:
            self.model = load_model(self.model_path)
            print(f"✅ Modelo carregado: {self.model_path}")
        except Exception as e:
            raise ValueError(f"❌ Erro ao carregar modelo: {str(e)}")
            
    def preprocess_image(self, image_path):
        """
        Pré-processa imagem para predição
        
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
        
        # Adiciona dimensão do batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    def predict_single_patch(self, image_path, return_probs=False):
        """
        Prediz classe de um único patch
        
        Args:
            image_path: Caminho da imagem
            return_probs: Se deve retornar probabilidades
            
        Returns:
            Predição (classe ou probabilidades)
        """
        # Pré-processa imagem
        img_array = self.preprocess_image(image_path)
        
        # Faz predição
        predictions = self.model.predict(img_array, verbose=0)
        
        if return_probs:
            return predictions[0]
        else:
            predicted_class = np.argmax(predictions[0])
            return predicted_class, self.class_names[predicted_class]
            
    def predict_patient_patches(self, patient_patches_dir, aggregation='majority_vote'):
        """
        Prediz classe para todos os patches de um paciente
        
        Args:
            patient_patches_dir: Diretório com patches do paciente
            aggregation: Método de agregação ('majority_vote', 'average_prob', 'max_confidence')
            
        Returns:
            Dicionário com resultados da predição
        """
        patches_dir = Path(patient_patches_dir)
        
        if not patches_dir.exists():
            raise ValueError(f"Diretório não encontrado: {patches_dir}")
            
        # Lista patches
        patch_files = list(patches_dir.glob("*.jpg"))
        
        if not patch_files:
            raise ValueError(f"Nenhum patch encontrado em: {patches_dir}")
            
        print(f"🔍 Analisando {len(patch_files)} patches...")
        
        # Predições para cada patch
        patch_predictions = []
        patch_probabilities = []
        patch_paths = []
        
        for patch_file in patch_files:
            try:
                # Predição do patch
                probs = self.predict_single_patch(patch_file, return_probs=True)
                pred_class = np.argmax(probs)
                
                patch_predictions.append(pred_class)
                patch_probabilities.append(probs)
                patch_paths.append(str(patch_file))
                
            except Exception as e:
                print(f"⚠️ Erro ao processar {patch_file}: {str(e)}")
                continue
                
        if not patch_predictions:
            raise ValueError("Nenhum patch foi processado com sucesso")
            
        # Agrega predições
        if aggregation == 'majority_vote':
            # Voto majoritário
            final_prediction = max(set(patch_predictions), key=patch_predictions.count)
            confidence = patch_predictions.count(final_prediction) / len(patch_predictions)
            
        elif aggregation == 'average_prob':
            # Média das probabilidades
            avg_probs = np.mean(patch_probabilities, axis=0)
            final_prediction = np.argmax(avg_probs)
            confidence = avg_probs[final_prediction]
            
        elif aggregation == 'max_confidence':
            # Patch com maior confiança
            max_conf_idx = np.argmax([np.max(probs) for probs in patch_probabilities])
            final_prediction = patch_predictions[max_conf_idx]
            confidence = np.max(patch_probabilities[max_conf_idx])
            
        else:
            raise ValueError(f"Método de agregação inválido: {aggregation}")
            
        results = {
            'patient_prediction': final_prediction,
            'patient_class': self.class_names[final_prediction],
            'confidence': confidence,
            'aggregation_method': aggregation,
            'num_patches': len(patch_predictions),
            'patch_predictions': patch_predictions,
            'patch_probabilities': patch_probabilities,
            'patch_paths': patch_paths,
            'class_distribution': {
                self.class_names[i]: patch_predictions.count(i) 
                for i in range(len(self.class_names))
            }
        }
        
        return results
        
    def predict_batch_patients(self, patients_base_dir, output_csv=None):
        """
        Prediz classes para múltiplos pacientes
        
        Args:
            patients_base_dir: Diretório base com subpastas de pacientes
            output_csv: Caminho para salvar resultados em CSV
            
        Returns:
            DataFrame com resultados
        """
        base_dir = Path(patients_base_dir)
        
        if not base_dir.exists():
            raise ValueError(f"Diretório não encontrado: {base_dir}")
            
        # Lista diretórios de pacientes
        patient_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        
        print(f"🏥 Processando {len(patient_dirs)} pacientes...")
        
        results_list = []
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            try:
                print(f"👤 Paciente {patient_id}...")
                
                # Prediz para o paciente
                patient_results = self.predict_patient_patches(patient_dir)
                
                # Adiciona ID do paciente
                patient_results['patient_id'] = patient_id
                results_list.append(patient_results)
                
                print(f"   ✅ {patient_results['patient_class']} "
                      f"(confiança: {patient_results['confidence']:.3f})")
                      
            except Exception as e:
                print(f"   ❌ Erro: {str(e)}")
                # Adiciona resultado com erro
                results_list.append({
                    'patient_id': patient_id,
                    'patient_prediction': -1,
                    'patient_class': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })
                continue
                
        # Converte para DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Salva em CSV se especificado
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            print(f"💾 Resultados salvos em: {output_csv}")
            
        return df_results
        
    def visualize_patient_results(self, patient_results, show_patches=True, max_patches=9):
        """
        Visualiza resultados de predição para um paciente
        
        Args:
            patient_results: Resultados da predição do paciente
            show_patches: Se deve mostrar imagens dos patches
            max_patches: Máximo de patches a mostrar
        """
        
        print("="*60)
        print(f"📊 RESULTADOS DA PREDIÇÃO")
        print("="*60)
        print(f"🎯 Predição final: {patient_results['patient_class']}")
        print(f"📈 Confiança: {patient_results['confidence']:.3f}")
        print(f"🔢 Total de patches: {patient_results['num_patches']}")
        print(f"⚙️ Método de agregação: {patient_results['aggregation_method']}")
        
        print(f"\n📊 Distribuição das predições por patch:")
        for class_name, count in patient_results['class_distribution'].items():
            percentage = count / patient_results['num_patches'] * 100
            print(f"   {class_name}: {count} patches ({percentage:.1f}%)")
            
        # Visualiza patches se solicitado
        if show_patches and 'patch_paths' in patient_results:
            self._plot_patch_predictions(patient_results, max_patches)
            
    def _plot_patch_predictions(self, patient_results, max_patches):
        """Plota patches com suas predições"""
        
        patch_paths = patient_results['patch_paths']
        patch_predictions = patient_results['patch_predictions']
        patch_probabilities = patient_results['patch_probabilities']
        
        # Limita número de patches a mostrar
        n_show = min(len(patch_paths), max_patches)
        
        # Calcula layout do grid
        cols = 3
        rows = (n_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
        
        for i in range(n_show):
            patch_path = patch_paths[i]
            prediction = patch_predictions[i]
            probabilities = patch_probabilities[i]
            
            # Carrega e mostra imagem
            img = cv2.imread(patch_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'Patch {i+1}\n{self.class_names[prediction]}\n'
                             f'Conf: {probabilities[prediction]:.3f}')
            axes[i].axis('off')
            
        # Remove axes extras
        for i in range(n_show, len(axes)):
            axes[i].remove()
            
        plt.tight_layout()
        plt.show()

# Funções utilitárias
def predict_single_patient(model_path, patient_patches_dir):
    """
    Função utilitária para predizer um único paciente
    
    Args:
        model_path: Caminho do modelo treinado
        patient_patches_dir: Diretório com patches do paciente
        
    Returns:
        Resultados da predição
    """
    predictor = InceptionALNPredictor(model_path)
    results = predictor.predict_patient_patches(patient_patches_dir)
    predictor.visualize_patient_results(results)
    
    return results

def predict_all_patients(model_path, patients_base_dir, output_csv="predictions.csv"):
    """
    Função utilitária para predizer todos os pacientes
    
    Args:
        model_path: Caminho do modelo treinado
        patients_base_dir: Diretório base com pacientes
        output_csv: Arquivo CSV de saída
        
    Returns:
        DataFrame com resultados
    """
    predictor = InceptionALNPredictor(model_path)
    results_df = predictor.predict_batch_patients(patients_base_dir, output_csv)
    
    # Exibe resumo
    print("\n" + "="*60)
    print("📋 RESUMO DAS PREDIÇÕES")
    print("="*60)
    
    successful_predictions = results_df[results_df['patient_class'] != 'ERROR']
    
    if len(successful_predictions) > 0:
        print(f"✅ Pacientes processados com sucesso: {len(successful_predictions)}")
        print(f"❌ Pacientes com erro: {len(results_df) - len(successful_predictions)}")
        
        print(f"\n📊 Distribuição das predições:")
        pred_counts = successful_predictions['patient_class'].value_counts()
        for class_name, count in pred_counts.items():
            percentage = count / len(successful_predictions) * 100
            print(f"   {class_name}: {count} pacientes ({percentage:.1f}%)")
            
        print(f"\n📈 Confiança média: {successful_predictions['confidence'].mean():.3f}")
        
    return results_df

# Função principal para execução
def main():
    """Função principal com interface de linha de comando"""
    
    print("🔬 PREDITOR INCEPTION V4 - CLASSIFICAÇÃO ALN")
    print("="*50)
    
    # Busca modelo treinado
    base_dir = Path(__file__).parent
    model_candidates = [
        base_dir / "best_inception_aln_model.h5",
        base_dir / "inception_v4_aln_final.h5"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break
            
    if not model_path:
        model_path = input("Digite o caminho do modelo (.h5): ").strip()
        if not Path(model_path).exists():
            print(f"❌ Modelo não encontrado: {model_path}")
            return
            
    print(f"📁 Usando modelo: {model_path}")
    
    # Menu de opções
    print("\nEscolha o modo:")
    print("1. Predizer um paciente específico")
    print("2. Predizer todos os pacientes")
    
    choice = input("\nDigite sua escolha (1-2): ").strip()
    
    if choice == "1":
        patient_dir = input("Digite o diretório do paciente: ").strip()
        results = predict_single_patient(model_path, patient_dir)
        
    elif choice == "2":
        patients_dir = input("Digite o diretório base dos pacientes: ").strip()
        output_file = input("Arquivo CSV de saída (Enter=predictions.csv): ").strip()
        if not output_file:
            output_file = "predictions.csv"
            
        results_df = predict_all_patients(model_path, patients_dir, output_file)
        
    else:
        print("❌ Opção inválida")

if __name__ == "__main__":
    main()
