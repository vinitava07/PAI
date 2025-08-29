from typing import Dict
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import gc

# Define caminhos
base_dir = Path(__file__).parent.parent
patches_dir = base_dir / "patches"
clinical_data_path = base_dir / "patient-clinical-data.csv"

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

def check_cuda_availability():
    """Verifica se CUDA está disponível no XGBoost"""
    try:
        # Tenta criar um modelo XGBoost com GPU
        test_model = xgb.XGBClassifier(tree_method='hist', device='cuda')
        print("✓ CUDA disponível no XGBoost")
        return True
    except Exception as e:
        print(f"✗ CUDA não disponível no XGBoost: {e}")
        print("  Usando CPU (tree_method='hist')")
        return False

def build_feature_extractor(input_size: int):
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(input_size, input_size, 3), pooling='avg')
    return base_model

def extract_features_from_df(df: pd.DataFrame, model, input_size: int, batch_size: int = 16):

    paths = df['patch_path'].tolist()
    labels = df['class'].astype(int).tolist()
    n = len(paths)
    features_list = []
    
    print(f"Extraindo features de {n} imagens (batch_size={batch_size})...")
    
    # Processar em batches
    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        batch_imgs = []
        
        # Carrega imagens do batch
        for p in batch_paths:
            try:
                # Carrega imagem e redimensiona
                img = image.load_img(p, target_size=(input_size, input_size))
                x = image.img_to_array(img)
                batch_imgs.append(x)
            except Exception as e:
                print(f"Erro ao carregar imagem {p}: {e}")
                # Cria imagem vazia em caso de erro
                x = np.zeros((input_size, input_size, 3))
                batch_imgs.append(x)
        
        if batch_imgs:
            batch_array = np.stack(batch_imgs, axis=0)
            batch_pre = resnet_preprocess(batch_array)
            feats = model.predict(batch_pre, verbose=0)  # shape (batch_size, n_features)
            features_list.append(feats)
            
            # Libera memória
            del batch_array, batch_pre, batch_imgs
            
            if (i + 1) % 10 == 0:
                print(f"  Processados {end}/{n} patches...")
                gc.collect()
    
    if features_list:
        X = np.vstack(features_list)
        y = np.array(labels)
        print(f"Features extraídas: {X.shape}")
        return X, y
    else:
        raise ValueError("Nenhuma feature foi extraída!")

def get_xgb_params(use_gpu=False):

    base_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }
    
    if use_gpu:
        base_params.update({
            'tree_method': 'hist',
            'device': 'cuda',
            'max_bin': 16,  # Reduzido para GTX 1660
        })
    else:
        base_params.update({
            'tree_method': 'hist',
            'device': 'cpu',
            'n_jobs': -1,
        })
    
    return base_params

def train_xgb_classifier(X_train, y_train, X_val, y_val, param_grid=None, use_gpu=False):
    """
    Treina XGBClassifier otimizado para GTX 1660.
    """
    print(f"\nTreinando XGBoost ({'GPU' if use_gpu else 'CPU'})...")
    
    if param_grid:
        # Grid search com parâmetros base
        base_params = get_xgb_params(use_gpu)
        xgb_clf = xgb.XGBClassifier(**base_params)
        
        # Ajusta n_jobs para não sobrecarregar
        cv_jobs = 2 if use_gpu else -1  # Limita jobs quando usando GPU
        
        gs = GridSearchCV(
            xgb_clf, 
            param_grid, 
            cv=3, 
            scoring='accuracy', 
            verbose=1, 
            n_jobs=cv_jobs,
            error_score='raise'
        )
        
        try:
            gs.fit(X_train, y_train)
            print("Melhores parâmetros XGBoost:", gs.best_params_)
            best = gs.best_estimator_
        except Exception as e:
            print(f"Erro no GridSearch: {e}")
            print("Usando parâmetros padrão...")
            best = xgb.XGBClassifier(**get_xgb_params(use_gpu))
            best.fit(X_train, y_train)
    else:
        # Treina com parâmetros otimizados
        params = get_xgb_params(use_gpu)
        best = xgb.XGBClassifier(**params)
        
        try:
            best.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        except Exception as e:
            print(f"Erro durante treinamento: {e}")
            # Treina sem early stopping em caso de erro
            best.fit(X_train, y_train)
    
    # Avaliar no conjunto de validação
    try:
        y_pred_val = best.predict(X_val)
        print("\nRelatório de classificação no conjunto de validação:")
        print(classification_report(y_val, y_pred_val))
        print("\nMatriz de confusão (val):")
        print(confusion_matrix(y_val, y_pred_val))
    except Exception as e:
        print(f"Erro na avaliação: {e}")
    
    return best

def evaluate_on_test(model, X_test, y_test):
    """Avalia modelo no conjunto de teste"""
    try:
        y_pred = model.predict(X_test)
        print("\n" + "="*50)
        print("AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("="*50)
        print(classification_report(y_test, y_pred))
        print("\nMatriz de confusão (teste):")
        print(confusion_matrix(y_test, y_pred))
        
        # Calcula acurácia por classe
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia geral: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Erro na avaliação de teste: {e}")

# Exemplo de uso otimizado
if __name__ == "__main__":
    try:
        # Verifica disponibilidade de CUDA
        use_gpu = check_cuda_availability()
        
        # Inicializa classificador
        classifier = BaseALNClassifier(
            patches_dir=str(patches_dir),
            clinical_data_path=str(clinical_data_path)
        )
        
        # Prepara dataset
        datasets = classifier.prepare_dataset(test_size=0.2)
        train_df = datasets['train']
        val_df = datasets['val']
        test_df = datasets['test']
        
        # Tamanho de entrada reduzido para GTX 1660
        input_size = 224  # Reduzido de 256 para 224
        
        print(f"\nConstruindo extrator de features (input_size={input_size})...")
        feature_model = build_feature_extractor(input_size)
        
        # Extrai features com batch_size reduzido
        print("\nExtraindo features de treino...")
        X_train, y_train = extract_features_from_df(train_df, feature_model, input_size, batch_size=16)
        
        print("Extraindo features de validação...")
        X_val, y_val = extract_features_from_df(val_df, feature_model, input_size, batch_size=16)
        
        print("Extraindo features de teste...")
        X_test, y_test = extract_features_from_df(test_df, feature_model, input_size, batch_size=16)
        
        # Libera modelo de features da memória
        del feature_model
        gc.collect()
        
        # Grid de parâmetros simplificado para GTX 1660
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        # Treina modelo
        model = train_xgb_classifier(
            X_train, y_train, X_val, y_val, 
            param_grid=param_grid,  # Use None para treino mais rápido
            use_gpu=use_gpu
        )
        
        # Avalia no teste
        evaluate_on_test(model, X_test, y_test)
        
        # Salva modelo
        model_path = 'xgb_model.pkl'
        joblib.dump(model, model_path)
        print(f"\nModelo salvo em: {model_path}")
        
    except Exception as e:
        print(f"\nERRO GERAL: {e}")
        import traceback
        traceback.print_exc()