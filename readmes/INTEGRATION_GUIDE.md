# 🔬 Guia de Integração: Dados Clínicos + Features Morfológicos para XGBoost

## 📋 Resumo da Integração

Com base na análise dos seus arquivos, criei um sistema completo que integra:

- **1.058 pacientes** do arquivo CSV com dados clínicos
- **Features morfológicos** extraídos das imagens usando seu `segmentation.py`
- **Pipeline XGBoost** para classificação de metástase linfonodal

---

## 🗂️ Estrutura de Dados Identificada

### 1. **Dados Clínicos (CSV)**
```
Patient ID | Age | Tumour Size | ER | PR | HER2 | Molecular subtype | ALN status
1          | 77  | 3           | +  | +  | -    | Luminal A         | N0
2          | 39  | 3,5         | -  | -  | -    | Triple negative   | N+(>2)
...
```

**Classes Target:**
- `N0`: Sem metástase linfonodal
- `N+(1-2)`: 1-2 linfonodos positivos  
- `N+(>2)`: Mais de 2 linfonodos positivos

### 2. **Patches de Imagens**
```
paper_patches/patches/
├── 1/          # Paciente 1
│   ├── 1_0_0_0.jpg
│   ├── 1_0_0_256.jpg
│   └── ...
├── 2/          # Paciente 2
│   ├── 2_0_256_768.jpg
│   └── ...
```

### 3. **Features Extraídos (segmentation.py)**
- **Área**: Tamanho dos núcleos
- **Circularidade**: Regularidade da forma
- **Excentricidade**: Alongamento do núcleo
- **Distância normalizada**: Distância ao vizinho mais próximo / raio

---

## 🔧 Como Usar o Sistema

### **Passo 1: Processar Todas as Imagens**
```python
from xgboost_integration import BreastCancerDataIntegration

# Inicializa integrador
integrator = BreastCancerDataIntegration(
    csv_path="patient-clinical-data.csv",
    patches_base_path="paper_patches/patches"
)

# Processa todos os pacientes (pode demorar)
integrator.process_all_patients(
    max_patients=None,        # None = todos os pacientes
    max_patches_per_patient=5 # Limita patches por paciente
)

# Integra dados clínicos + morfológicos
dataset = integrator.integrate_data()

# Prepara para XGBoost
X, y, feature_names = integrator.prepare_xgboost_data()

# Exporta dataset final
integrator.export_dataset("breast_cancer_complete.csv")
```

### **Passo 2: Treinar XGBoost**
```python
from xgboost_training import XGBoostBreastCancerClassifier

# Inicializa classificador
classifier = XGBoostBreastCancerClassifier()

# Prepara dados
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)

# Treina modelo
classifier.train_model(X_train, y_train)

# Avalia resultados
results = classifier.evaluate_model(X_test, y_test)

# Gera visualizações
classifier.plot_feature_importance(feature_names)
classifier.plot_confusion_matrix(results['confusion_matrix'])
classifier.generate_scatter_plots(X, y, feature_names)
```

---

## 📊 Features Finais para XGBoost

### **Dados Clínicos (7 features)**
- `Age(years)`: Idade do paciente
- `Tumor_Size_Numeric`: Tamanho do tumor (cm)
- `ER_encoded`: Status receptor estrogênio
- `PR_encoded`: Status receptor progesterona
- `HER2_encoded`: Status HER2
- `Tumour Type_encoded`: Tipo histológico
- `Molecular subtype_encoded`: Subtipo molecular

### **Features Morfológicos (22 features)**
**Características Principais (especificadas no projeto):**
- `area_mean`, `area_std`: Área dos núcleos
- `circularity_mean`, `circularity_std`: Circularidade 
- `eccentricity_mean`, `eccentricity_std`: Excentricidade
- `normalized_nn_distance_mean`, `normalized_nn_distance_std`: Distância normalizada

**Características Complementares:**
- `num_nuclei`: Número total de núcleos
- `nuclear_density`: Densidade nuclear
- `solidity_mean`, `solidity_std`: Solidez dos núcleos
- `orientation_std`: Variabilidade da orientação
- `area_p25`, `area_p75`: Percentis de área
- `circularity_p25`, `circularity_p75`: Percentis de circularidade
- `large_nuclei_ratio`: Proporção de núcleos grandes
- `irregular_nuclei_ratio`: Proporção de núcleos irregulares
- `eccentric_nuclei_ratio`: Proporção de núcleos excêntricos

---

## 🎯 Funcionalidades Implementadas

### ✅ **Extração de Features**
- Processa todos os patches de cada paciente
- Agrega características por paciente (média, desvio, percentis)
- Trata pacientes sem imagens (features = 0)

### ✅ **Integração de Dados**
- Merge automático por Patient ID
- Encoding categórico das variáveis clínicas
- Tratamento de valores faltantes

### ✅ **Modelo XGBoost**
- Parâmetros otimizados para dados médicos
- Balanceamento de classes automático
- Validação cruzada estratificada

### ✅ **Visualizações**
- Gráficos de dispersão (conforme projeto)
- Importância das features
- Matriz de confusão

---

## 🚀 Próximos Passos Sugeridos

### **1. Processamento Completo**
```bash
# Execute o pipeline completo
python xgboost_integration.py  # Processa todas as imagens
python xgboost_training.py     # Treina modelo
```

### **2. Validação dos Resultados**
- Compare features extraídos com literatura médica
- Analise correlações entre características morfológicas e classes
- Valide se núcleos estão sendo segmentados corretamente

### **3. Otimização do Modelo**
- Hyperparameter tuning com Optuna
- Teste diferentes agregações de features
- Cross-validation com diferentes splits

### **4. Implementação dos Classificadores Profundos**
- Inception para patches individuais
- MobileNet como feature extractor
- Compare performance com XGBoost

---

## ⚠️ Considerações Importantes

### **Qualidade da Segmentação**
- Verifique visualmente alguns resultados do `segmentation.py`
- Ajuste parâmetros se necessário
- Considere validation manual de alguns casos

### **Balanceamento de Classes**
```
N0: ~655 pacientes (62%)
N+(1-2): ~210 pacientes (20%) 
N+(>2): ~193 pacientes (18%)
```

### **Processamento Computacional**
- ~1.058 pacientes × múltiplos patches = muito processamento
- Considere processamento em lotes
- Use cache para evitar reprocessamento

### **Separação Treino/Teste**
- **CRÍTICO**: Separe por paciente, não por patch
- Garanta que patches do mesmo paciente não apareçam em treino e teste
- Use stratified split para manter proporção das classes

---

## 📝 Exemplo de Execução Rápida

```python
# Para testar rapidamente com poucos pacientes
integrator = BreastCancerDataIntegration(csv_path, patches_path)

# Apenas 20 pacientes, 2 patches cada
integrator.process_all_patients(max_patients=20, max_patches_per_patient=2)
dataset = integrator.integrate_data()
X, y, features = integrator.prepare_xgboost_data()

# Treina modelo
classifier = XGBoostBreastCancerClassifier()
X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
classifier.train_model(X_train, y_train)
results = classifier.evaluate_model(X_test, y_test)

print(f"Acurácia: {results['accuracy']:.4f}")
```

Este sistema fornece uma base sólida para seu projeto, integrando perfeitamente os dados clínicos com as características morfológicas extraídas das imagens!
