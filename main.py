import os
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from interface import ImageViewer


filepath = 'patient-clinical-data.xlsx'
try:
    df = pd.read_excel(filepath)

    print("first five lines: ")
    print(df.head())
except FileNotFoundError:
    print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")
base_dir = "paper_patches/patches" # <<< ALTERE AQUI SE NECESSÁRIO
if not os.path.isdir(base_dir):
    # Tentar encontrar a pasta subindo um nível (caso o script esteja dentro da pasta do projeto)
    alt_base_dir = os.path.join("..", base_dir)
    if os.path.isdir(alt_base_dir):
        base_dir = alt_base_dir
    else:
        # Última tentativa: pedir ao usuário
        messagebox.showerror("Erro de Configuração",
                             f"A pasta '{base_dir}' não foi encontrada. "
                             "Por favor, verifique a variável 'base_dir' no código "
                             "ou selecione a pasta 'patches' manualmente.")
        chosen_dir = filedialog.askdirectory(title="Selecione a pasta 'patches' dentro de 'paper_patches'")
        if chosen_dir:
            base_dir = chosen_dir
        else:
            print("Nenhuma pasta selecionada. Saindo.")
            exit()

if not os.path.isdir(base_dir):
    messagebox.showerror("Erro Fatal", f"A pasta de patches '{base_dir}' não existe ou não é acessível. Verifique o caminho e tente novamente.")
    exit()
app = ImageViewer(base_path=base_dir, min_folder=1, max_folder=1058)
app.mainloop()
