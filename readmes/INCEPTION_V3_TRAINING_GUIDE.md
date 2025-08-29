# 🧠 Treinamento Inception V3 para Classificação ALN

Este documento explica como treinar o modelo Inception V3 para classificação de status de linfonodos axilares (ALN) em imagens histológicas de câncer de mama.

## 📋 Visão Geral

O sistema classifica patches de imagens histológicas em três categorias:
- **N0**: Sem metástase linfonodal
- **N+(1-2)**: 1-2 linfonodos com metástase
- **N+(>2)**: Mais de 2 linfonodos com metástase

## 🏗️ Arquitetura

- **Modelo Base**: Inception V3 pré-treinado no ImageNet
- **Estratégia**: Transfer Learning com fine-tuning em duas fases
- **Input**: Patches de 299x299 pixels
- **Output**: Probabilidades para 3 classes

## 📁 Estrutura de Dados

```
PAI/
├── patches/                    # Diretório com patches organizados por paciente
│   ├── 1/                     # Paciente 1
│   │   ├── 1_0_0_0.jpg       # Patches do paciente 1
│   │   ├── 1_0_0_256.jpg
│   │   └── ...
│   ├── 2/                     # Paciente 2
│   └── ...
├── patient-clinical-data.csv   # Dados clínicos com ALN status
└── models/                     # Diretório de saída (criado automaticamente)
```

## 🚀 Como Executar

### Opção 1: Menu Interativo (Recomendado)

```bash
python run_inception_training.py
```

Escolha uma das opções:
1. **Treinamento rápido** (5 patches/paciente, ~30 min)
2. **Treinamento médio** (10 patches/paciente, ~1-2 horas)
3. **Treinamento completo** (todos patches, ~4-6 horas)

### Opção 2: Linha de Comando

```bash
# Treinamento médio (recomendado)
python run_inception_training.py --mode medio

# Treinamento rápido para testes
python run_inception_training.py --mode rapido

# Treinamento customizado
python run_inception_training.py --mode custom --patches 15 --epochs 40 --batch-size 8
```

### Opção 3: Script Direto

```python
from train_inception_v3_improved import InceptionV3ALNClassifier

# Inicializar
classifier = InceptionV3ALNClassifier(
    patches_dir="patches",
    clinical_data_path="patient-clinical-data.csv"
)

# Carregar dados
classifier.load_clinical_data()

# Preparar dataset (80% treino, 20% teste)
datasets = classifier.prepare_dataset(
    max_patches_per_patient=10,  # ou None para todos
    test_size=0.2,
    val_size=0.1
)

# Construir modelo
model = classifier.build_model()

# Treinar
history = classifier.train_model(
    datasets=datasets,
    epochs=30,
    fine_tune_epochs=15
)

# Avaliar
results = classifier.evaluate_model(datasets['test'])
```

## ⚙️ Parâmetros Importantes

### Preparação dos Dados
- `max_patches_per_patient`: Limita patches por paciente (None = todos)
- `test_size`: Proporção de pacientes para teste (padrão: 0.2 = 20%)
- `val_size`: Proporção para validação do treino (padrão: 0.1 = 10%)

### Treinamento
- `epochs`: Épocas para fase 1 (feature extraction)
- `fine_tune_epochs`: Épocas para fase 2 (fine-tuning)
- `batch_size`: Tamanho do batch (reduzir se houver erro de memória)
- `fine_tune_layers`: Número de camadas a descongelar

## 📊 Divisão dos Dados

**IMPORTANTE**: A divisão é feita por **PACIENTE**, não por patch!

```
Total de pacientes: 1058
├── Treino: ~740 pacientes (70%)
├── Validação: ~106 pacientes (10%)
└── Teste: ~212 pacientes (20%)
```

Isso garante que:
- ✅ Não há vazamento de dados entre conjuntos
- ✅ Patches do mesmo paciente ficam no mesmo conjunto
- ✅ Avaliação realista da capacidade de generalização

## 🎯 Processo de Treinamento

### Fase 1: Feature Extraction
- Modelo base (Inception V3) congelado
- Treina apenas camadas de classificação
- Learning rate: 0.001
- Duração: ~30 épocas

### Fase 2: Fine-tuning
- Descongela últimas 100-150 camadas
- Learning rate reduzido: 0.00001
- Duração: ~15-20 épocas

## 📈 Métricas e Avaliação

O sistema calcula:
- **Por Patch**: Acurácia, Precisão, Recall, F1-Score, ROC AUC
- **Por Paciente**: Agregação das predições de todos os patches

### Arquivos de Saída

```
models/run_YYYYMMDD_HHMMSS/
├── best_model_final.h5          # Modelo treinado
├── training_curves_*.png        # Gráficos de treinamento
├── confusion_matrix.png         # Matriz de confusão
├── roc_curves.png              # Curvas ROC
├── test_metrics.json           # Métricas detalhadas
├── classification_report.csv    # Relatório por classe
├── test_predictions.csv        # Predições no teste
└── relatorio_final.html        # Relatório completo
```

## 🔧 Resolução de Problemas

### Erro de Memória GPU
```bash
# Reduzir batch size
python run_inception_training.py --mode medio --batch-size 8
```

### Processo Muito Lento
```bash
# Usar menos patches por paciente
python run_inception_training.py --mode custom --patches 5 --epochs 20
```

### Verificar GPU
```python
import tensorflow as tf
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
```

## 💡 Dicas de Uso

1. **Desenvolvimento**: Use modo 'rapido' para testar o pipeline
2. **Produção**: Use modo 'medio' ou 'completo'
3. **Memória**: Reduza batch_size se houver erro de GPU
4. **Patches**: Limite patches por paciente para treinos mais rápidos

## 📊 Resultados Esperados

Com configuração padrão (modo médio):
- Acurácia: ~75-85%
- AUC: ~0.85-0.92
- Tempo: 1-2 horas com GPU

## 🔍 Análise dos Resultados

Após o treinamento, abra o arquivo `relatorio_final.html` no navegador para visualizar:
- Métricas detalhadas por classe
- Curvas de treinamento
- Matriz de confusão
- Curvas ROC
- Avaliação por paciente

## 📝 Exemplo de Uso Completo

```bash
# 1. Verificar dados
ls patches/ | wc -l  # Deve mostrar 1058 diretórios

# 2. Executar treinamento médio
python run_inception_training.py --mode medio

# 3. Verificar resultados
cd models/run_*/
firefox relatorio_final.html
```

## ⚠️ Observações Importantes

1. **GPU Recomendada**: O treinamento é ~10x mais rápido com GPU
2. **Dados Balanceados**: O sistema usa class weights automáticos
3. **Reprodutibilidade**: Seeds fixas garantem resultados consistentes
4. **Validação**: Monitora overfitting com early stopping

---

Para mais informações ou problemas, verifique os logs de treinamento no diretório de saída.
