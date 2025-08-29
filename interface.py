"""
Interface Gráfica para Análise de Imagens Histológicas de Câncer de Mama
Desenvolvido para classificação de metástase em linfonodos axilares (ALN)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend não-interativo para evitar problemas com tkinter
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import subprocess
import pandas as pd
import os

# Adiciona os diretórios necessários ao path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "segmentation"))
sys.path.append(str(Path(__file__).parent / "xgboost"))

# Importa os módulos do projeto
from segmentation import HENucleusSegmentation
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


class BreastCancerAnalysisGUI:
    def __init__(self, root):
        """
        Inicializa a interface gráfica principal
        """
        self.root = root
        self.root.title("Análise de Imagens Histológicas - Câncer de Mama")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variáveis de estado
        self.current_image = None  # Imagem PIL original
        self.current_image_path = None  # Caminho da imagem carregada
        self.segmenter = HENucleusSegmentation()  # Instância do segmentador
        self.current_screen = "main"  # Tela atual
        self.zoom_level = 1.0  # Nível de zoom
        
        # Configuração de estilo
        self.setup_styles()
        
        # Cria a tela inicial
        self.create_main_menu()
        
    def setup_styles(self):
        """
        Configura estilos visuais da interface
        """
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botões principais
        style.configure('Main.TButton', 
                       font=('Arial', 14, 'bold'),
                       padding=(20, 15))
        
        # Estilo para botões secundários
        style.configure('Secondary.TButton',
                       font=('Arial', 11),
                       padding=(10, 8))
        
        # Estilo para labels
        style.configure('Title.TLabel',
                       font=('Arial', 24, 'bold'),
                       background='#f0f0f0')
        
        style.configure('Subtitle.TLabel',
                       font=('Arial', 12),
                       background='#f0f0f0')
        
    def clear_screen(self):
        """
        Limpa todos os widgets da tela atual
        """
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def create_main_menu(self):
        """
        Cria a tela principal com as três opções principais
        """
        self.clear_screen()
        self.current_screen = "main"
        
        # Container principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="Sistema de Análise de Imagens Histológicas",
                               style='Title.TLabel')
        title_label.pack(pady=(50, 10))
        
        subtitle_label = ttk.Label(main_frame,
                                 text="Classificação de Metástase em Linfonodos Axilares",
                                 style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 50))
        
        # Frame para botões
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack()
        
        # Botão 1: Treinar Modelos
        train_btn = ttk.Button(button_frame,
                             text="Treinar Modelos",
                             command=self.show_training_screen,
                             style='Main.TButton')
        train_btn.pack(pady=15)
        
        # Botão 2: Carregar e Analisar Imagem
        analyze_btn = ttk.Button(button_frame,
                               text="Carregar e Analisar Imagem",
                               command=self.show_image_selection_screen,
                               style='Main.TButton')
        analyze_btn.pack(pady=15)
        
        # Botão 3: Visualizar Estatísticas
        stats_btn = ttk.Button(button_frame,
                             text="Visualizar Estatísticas da Base",
                             command=self.show_statistics_screen,
                             style='Main.TButton')
        stats_btn.pack(pady=15)
        
        # Botão Sair
        exit_btn = ttk.Button(button_frame,
                            text="Sair",
                            command=self.root.quit,
                            style='Secondary.TButton')
        exit_btn.pack(pady=30)
        
    def create_back_button(self, parent, command=None):
        """
        Cria um botão de voltar padronizado
        
        Args:
            parent: Widget pai onde o botão será colocado
            command: Função a ser executada (padrão: voltar ao menu principal)
        """
        if command is None:
            command = self.create_main_menu
            
        back_btn = ttk.Button(parent,
                            text="← Voltar",
                            command=command,
                            style='Secondary.TButton')
        back_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
    def show_training_screen(self):
        """
        Exibe a tela de seleção de modelo para treinamento
        """
        self.clear_screen()
        self.current_screen = "training"
        
        # Container principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header com botão voltar
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill=tk.X)
        self.create_back_button(header_frame)
        
        # Título
        title_label = ttk.Label(main_frame,
                               text="Selecione o Modelo para Treinamento",
                               style='Title.TLabel')
        title_label.pack(pady=(30, 50))
        
        # Frame para botões de modelos
        models_frame = tk.Frame(main_frame, bg='#f0f0f0')
        models_frame.pack()
        
        # Informação sobre os modelos
        info_frame = tk.Frame(models_frame, bg='#f0f0f0')
        info_frame.pack(pady=(0, 30))
        
        info_text = """
        Escolha o modelo que deseja treinar:
        • XGBoost: Classificador raso baseado em características extraídas
        • Inception V3: Rede neural convolucional profunda
        • MobileNet V2: Rede neural otimizada para eficiência
        """
        
        info_label = tk.Label(info_frame, 
                            text=info_text,
                            font=('Arial', 11),
                            bg='#f0f0f0',
                            justify=tk.LEFT)
        info_label.pack()
        
        # Botões dos modelos
        xgboost_btn = ttk.Button(models_frame,
                               text="XGBoost",
                               command=lambda: self.train_model("xgboost"),
                               style='Main.TButton')
        xgboost_btn.pack(pady=10)
        
        inception_btn = ttk.Button(models_frame,
                                 text="Inception V3",
                                 command=lambda: self.train_model("inception"),
                                 style='Main.TButton')
        inception_btn.pack(pady=10)
        
        mobilenet_btn = ttk.Button(models_frame,
                                 text="MobileNet V2",
                                 command=lambda: self.train_model("mobilenet"),
                                 style='Main.TButton')
        mobilenet_btn.pack(pady=10)
        
    def train_model(self, model_type):
        """
        Inicia o treinamento do modelo selecionado
        
        Args:
            model_type: Tipo do modelo ('xgboost', 'inception', 'mobilenet')
        """
        # Confirma com o usuário
        response = messagebox.askyesno(
            "Confirmar Treinamento",
            f"Deseja iniciar o treinamento do modelo {model_type.upper()}?\n\n"
            "O progresso será exibido no console.\n"
            "A janela será fechada durante o treinamento."
        )
        
        if response:
            messagebox.showinfo(
                "Treinamento Iniciado",
                f"O treinamento do {model_type.upper()} foi iniciado.\n"
                "Verifique o console para acompanhar o progresso."
            )
            
            # Fecha a janela
            self.root.withdraw()
            
            # Inicia o treinamento no console
            if model_type == "xgboost":
                # Executa o script de treinamento do XGBoost
                subprocess.run([sys.executable, "xgboost/xgboost_training.py"])
            else:
                # Para Inception e MobileNet, usa o script unificado
                subprocess.run([sys.executable, "deep_models_unified.py", model_type])
            
            # Reabre a janela após o treinamento
            self.root.deiconify()
            self.create_main_menu()
            
    def show_image_selection_screen(self):
        """
        Exibe a tela de seleção e análise de imagem
        """
        self.clear_screen()
        self.current_screen = "image_analysis"
        
        # Container principal com duas colunas
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill=tk.X)
        self.create_back_button(header_frame)
        
        # Frame para controles (lado esquerdo)
        control_frame = tk.Frame(main_frame, bg='#f0f0f0', width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # Título dos controles
        control_title = ttk.Label(control_frame,
                                text="Controles de Análise",
                                font=('Arial', 16, 'bold'))
        control_title.pack(pady=(10, 20))
        
        # Botão para selecionar imagem
        select_btn = ttk.Button(control_frame,
                              text="Selecionar Imagem",
                              command=self.select_image,
                              style='Secondary.TButton')
        select_btn.pack(pady=10)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        # Label de funcionalidades
        func_label = ttk.Label(control_frame,
                             text="Funcionalidades:",
                             font=('Arial', 12, 'bold'))
        func_label.pack(pady=(10, 5))
        
        # Botões de funcionalidades (inicialmente desabilitados)
        self.grayscale_btn = ttk.Button(control_frame,
                                       text="Escala de Cinza",
                                       command=self.show_grayscale,
                                       state=tk.DISABLED,
                                       style='Secondary.TButton')
        self.grayscale_btn.pack(pady=5)
        
        self.segment_btn = ttk.Button(control_frame,
                                    text="Segmentação e Estatísticas",
                                    command=self.segment_image,
                                    state=tk.DISABLED,
                                    style='Secondary.TButton')
        self.segment_btn.pack(pady=5)
        
        self.classify_btn = ttk.Button(control_frame,
                                     text="Classificação",
                                     command=self.show_classification_menu,
                                     state=tk.DISABLED,
                                     style='Secondary.TButton')
        self.classify_btn.pack(pady=5)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        # Controles de zoom
        zoom_label = ttk.Label(control_frame,
                             text="Controle de Zoom:",
                             font=('Arial', 12, 'bold'))
        zoom_label.pack(pady=(10, 5))
        
        zoom_frame = tk.Frame(control_frame, bg='#f0f0f0')
        zoom_frame.pack(pady=10)
        
        zoom_in_btn = ttk.Button(zoom_frame,
                               text="Zoom +",
                               command=lambda: self.zoom_image(1.2))
        zoom_in_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_out_btn = ttk.Button(zoom_frame,
                                text="Zoom -",
                                command=lambda: self.zoom_image(0.8))
        zoom_out_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_reset_btn = ttk.Button(zoom_frame,
                                  text="Reset",
                                  command=lambda: self.zoom_image(1.0, reset=True))
        zoom_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame para exibição de imagens (lado direito)
        self.image_frame = tk.Frame(main_frame, bg='white')
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas com scrollbars para as imagens
        self.create_scrollable_canvas()
        
    def create_scrollable_canvas(self):
        """
        Cria um canvas com scrollbars para exibir imagens grandes
        """
        # Frame para o canvas e scrollbars
        canvas_frame = tk.Frame(self.image_frame, bg='white')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configura o canvas
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Frame interno para conteúdo
        self.canvas_content = tk.Frame(self.canvas, bg='white')
        self.canvas_window = self.canvas.create_window(0, 0, anchor=tk.NW, window=self.canvas_content)
        
        # Bind para atualizar scrollregion
        self.canvas_content.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        
    def select_image(self):
        """
        Abre diálogo para seleção de imagem
        """
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem de patch",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path):
        """
        Carrega e exibe a imagem selecionada
        
        Args:
            file_path: Caminho da imagem
        """
        try:
            # Carrega a imagem
            self.current_image_path = file_path
            self.current_image = Image.open(file_path)
            self.zoom_level = 1.0
            
            # Habilita os botões de funcionalidades
            self.grayscale_btn.config(state=tk.NORMAL)
            self.segment_btn.config(state=tk.NORMAL)
            self.classify_btn.config(state=tk.NORMAL)
            
            # Exibe a imagem
            self.display_image()
            
            # Exibe informações da imagem
            messagebox.showinfo(
                "Imagem Carregada",
                f"Imagem carregada com sucesso!\n\n"
                f"Arquivo: {Path(file_path).name}\n"
                f"Dimensões: {self.current_image.size[0]}x{self.current_image.size[1]} pixels"
            )
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")
            
    def display_image(self, second_image=None):
        """
        Exibe a imagem no canvas com zoom aplicado
        
        Args:
            second_image: Imagem adicional para exibir ao lado (opcional)
        """
        # Limpa o canvas
        for widget in self.canvas_content.winfo_children():
            widget.destroy()
            
        # Frame para as imagens
        images_frame = tk.Frame(self.canvas_content, bg='white')
        images_frame.pack(padx=20, pady=20)
        
        # Calcula o tamanho com zoom
        if self.current_image:
            width = int(self.current_image.width * self.zoom_level)
            height = int(self.current_image.height * self.zoom_level)
            
            # Redimensiona a imagem principal
            resized_image = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            
            # Label para imagem principal
            img_label = tk.Label(images_frame, image=photo, bg='white')
            img_label.image = photo  # Mantém referência
            img_label.pack(side=tk.LEFT, padx=10)
            
            # Se houver segunda imagem
            if second_image is not None:
                # Redimensiona a segunda imagem
                if isinstance(second_image, np.ndarray):
                    # Converte numpy array para PIL Image
                    if len(second_image.shape) == 2:  # Grayscale
                        second_pil = Image.fromarray(second_image)
                    else:  # Color
                        second_pil = Image.fromarray(cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB))
                else:
                    second_pil = second_image
                    
                resized_second = second_pil.resize((width, height), Image.Resampling.LANCZOS)
                photo2 = ImageTk.PhotoImage(resized_second)
                
                # Label para segunda imagem
                img_label2 = tk.Label(images_frame, image=photo2, bg='white')
                img_label2.image = photo2  # Mantém referência
                img_label2.pack(side=tk.LEFT, padx=10)
                
    def zoom_image(self, factor, reset=False):
        """
        Aplica zoom na imagem
        
        Args:
            factor: Fator de zoom
            reset: Se True, reseta o zoom para 1.0
        """
        if self.current_image is None:
            return
            
        if reset:
            self.zoom_level = 1.0
        else:
            # Limita o zoom entre 0.1 e 5.0
            new_zoom = self.zoom_level * factor
            if 0.1 <= new_zoom <= 5.0:
                self.zoom_level = new_zoom
                
        # Reexibe a imagem com novo zoom
        self.display_image()
        
    def show_grayscale(self):
        """
        Exibe a imagem em escala de cinza com canal de hematoxilina
        """
        if self.current_image_path is None:
            return
            
        try:
            # Processa a imagem para obter o canal de hematoxilina
            self.segmenter.preprocess_he_image(self.current_image_path)
            
            # Exibe a imagem original e o canal de hematoxilina lado a lado
            self.display_image(second_image=self.segmenter.hematoxylin_channel)
            
            messagebox.showinfo(
                "Escala de Cinza",
                "Canal de hematoxilina (núcleos) extraído e exibido ao lado da imagem original."
            )
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem:\n{str(e)}")
            
    def segment_image(self):
        """
        Realiza a segmentação da imagem e exibe os resultados
        """
        if self.current_image_path is None:
            return
            
        try:
            # Mensagem de processamento
            messagebox.showinfo(
                "Processando",
                "A segmentação está sendo realizada.\n"
                "Os resultados serão exibidos em uma nova janela."
            )
            
            # Desabilita temporariamente o matplotlib interativo
            plt.ioff()
            
            # Realiza a segmentação com visualização
            features, stats, labeled = self.segmenter.process_image(
                self.current_image_path,
                visualize=True
            )
            
            # Fecha todas as figuras matplotlib
            plt.close('all')
            
            # Exibe estatísticas resumidas
            if stats:
                stats_text = f"""
                Segmentação concluída com sucesso!
                
                Total de núcleos detectados: {stats['num_nuclei']}
                
                Área média: {stats['area']['mean']:.2f} ± {stats['area']['std']:.2f} pixels
                Circularidade média: {stats['circularity']['mean']:.3f} ± {stats['circularity']['std']:.3f}
                Excentricidade média: {stats['eccentricity']['mean']:.3f} ± {stats['eccentricity']['std']:.3f}
                Distância NN normalizada: {stats['normalized_nn_distance']['mean']:.3f} ± {stats['normalized_nn_distance']['std']:.3f}
                """
                
                messagebox.showinfo("Estatísticas de Segmentação", stats_text)
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na segmentação:\n{str(e)}")
            
    def show_classification_menu(self):
        """
        Exibe menu para seleção do modelo de classificação
        """
        # Cria janela de diálogo
        dialog = tk.Toplevel(self.root)
        dialog.title("Selecione o Modelo de Classificação")
        dialog.geometry("400x300")
        dialog.configure(bg='#f0f0f0')
        
        # Centraliza a janela
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Título
        title_label = ttk.Label(dialog,
                               text="Escolha o Modelo",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        # Frame para botões
        btn_frame = tk.Frame(dialog, bg='#f0f0f0')
        btn_frame.pack(pady=20)
        
        # Botões dos modelos
        models = [
            ("XGBoost", "xgboost"),
            ("Inception V3", "inception"),
            ("MobileNet V2", "mobilenet")
        ]
        
        for model_name, model_type in models:
            btn = ttk.Button(btn_frame,
                           text=model_name,
                           command=lambda mt=model_type: self.classify_image(mt, dialog),
                           style='Secondary.TButton')
            btn.pack(pady=10)
            
        # Botão cancelar
        cancel_btn = ttk.Button(dialog,
                              text="Cancelar",
                              command=dialog.destroy)
        cancel_btn.pack(pady=10)
        
    def classify_image(self, model_type, dialog):
        """
        Classifica a imagem usando o modelo selecionado
        
        Args:
            model_type: Tipo do modelo ('xgboost', 'inception', 'mobilenet')
            dialog: Janela de diálogo para fechar
        """
        dialog.destroy()
        
        if self.current_image_path is None:
            return
            
        try:
            # Carrega o modelo apropriado
            model_path = Path(__file__).parent / "models"
            
            if model_type == "xgboost":
                # Para XGBoost, primeiro precisa extrair features
                messagebox.showinfo(
                    "Processando",
                    "Extraindo características da imagem para classificação..."
                )
                
                # Extrai features usando o segmentador
                features, stats, _ = self.segmenter.process_image(
                    self.current_image_path,
                    visualize=False
                )
                
                # Carrega o modelo XGBoost
                with open(model_path / "xgboost_model.pkl", 'rb') as f:
                    model = pickle.load(f)
                
                # Prepara features para classificação
                feature_vector = np.array([
                    stats['area']['mean'],
                    stats['area']['std'],
                    stats['circularity']['mean'],
                    stats['circularity']['std'],
                    stats['eccentricity']['mean'],
                    stats['eccentricity']['std'],
                    stats['normalized_nn_distance']['mean'],
                    stats['normalized_nn_distance']['std']
                ]).reshape(1, -1)
                
                # Faz a predição
                prediction = model.predict(feature_vector)[0]
                probabilities = model.predict_proba(feature_vector)[0]
                
            else:
                # Para modelos de deep learning
                messagebox.showinfo(
                    "Processando",
                    f"Classificando imagem com {model_type.upper()}..."
                )
                
                # Define o tamanho de entrada baseado no modelo
                if model_type == "inception":
                    input_size = (299, 299)
                    model_file = "inception_model.h5"
                else:  # mobilenet
                    input_size = (224, 224)
                    model_file = "mobilenet_model.h5"
                
                # Carrega e pré-processa a imagem
                img = cv2.imread(self.current_image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_size)
                img = np.expand_dims(img, axis=0)
                
                # Normaliza baseado no modelo
                if model_type == "inception":
                    img = tf.keras.applications.inception_v3.preprocess_input(img)
                else:
                    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                
                # Carrega o modelo
                model = load_model(model_path / model_file)
                
                # Faz a predição
                probabilities = model.predict(img)[0]
                prediction = np.argmax(probabilities)
                
            # Classes
            classes = ['N0', 'N+(1-2)', 'N+(>2)']
            predicted_class = classes[prediction]
            
            # Exibe resultado
            result_text = f"""
            Classificação concluída!
            
            Modelo utilizado: {model_type.upper()}
            
            Classe predita: {predicted_class}
            
            Probabilidades:
            • N0: {probabilities[0]:.2%}
            • N+(1-2): {probabilities[1]:.2%}
            • N+(>2): {probabilities[2]:.2%}
            """
            
            messagebox.showinfo("Resultado da Classificação", result_text)
            
        except FileNotFoundError:
            messagebox.showerror(
                "Erro",
                f"Modelo {model_type} não encontrado!\n\n"
                "Certifique-se de treinar o modelo antes de usar a classificação."
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na classificação:\n{str(e)}")
            
    def show_statistics_screen(self):
        """
        Exibe a tela de estatísticas da base de dados
        """
        self.clear_screen()
        self.current_screen = "statistics"
        
        # Container principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#f0f0f0')
        header_frame.pack(fill=tk.X)
        self.create_back_button(header_frame)
        
        # Título
        title_label = ttk.Label(main_frame,
                               text="Estatísticas da Base de Dados",
                               style='Title.TLabel')
        title_label.pack(pady=(30, 20))
        
        try:
            # Tenta carregar a imagem de estatísticas gerada pela segmentação
            stats_image_path = Path(__file__).parent / "estatisticas.png"
            
            if stats_image_path.exists():
                # Carrega e exibe a imagem
                stats_image = Image.open(stats_image_path)
                
                # Redimensiona se necessário
                max_width = 1000
                max_height = 600
                stats_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(stats_image)
                
                # Label para exibir a imagem
                img_label = tk.Label(main_frame, image=photo, bg='#f0f0f0')
                img_label.image = photo  # Mantém referência
                img_label.pack(pady=20)
                
            else:
                # Se não houver imagem, gera estatísticas básicas
                self.generate_basic_statistics(main_frame)
                
        except Exception as e:
            error_label = ttk.Label(main_frame,
                                  text=f"Erro ao carregar estatísticas:\n{str(e)}",
                                  font=('Arial', 12))
            error_label.pack(pady=50)
            
    def generate_basic_statistics(self, parent):
        """
        Gera e exibe estatísticas básicas da base de dados
        
        Args:
            parent: Widget pai onde as estatísticas serão exibidas
        """
        # Frame para estatísticas
        stats_frame = tk.Frame(parent, bg='white', relief=tk.RAISED, bd=2)
        stats_frame.pack(padx=50, pady=20, fill=tk.BOTH, expand=True)
        
        # Tenta ler dados clínicos
        try:
            clinical_data = pd.read_csv("patient-clinical-data.csv")
            
            # Desabilita o matplotlib interativo
            plt.ioff()
            
            # Cria figura matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Análise Estatística da Base de Dados', fontsize=16)
            
            # 1. Distribuição de classes ALN
            aln_counts = clinical_data['ALN status'].value_counts()
            axes[0, 0].bar(aln_counts.index, aln_counts.values)
            axes[0, 0].set_title('Distribuição de Classes ALN')
            axes[0, 0].set_xlabel('Classe ALN')
            axes[0, 0].set_ylabel('Número de Pacientes')
            
            # 2. Distribuição de idade
            axes[0, 1].hist(clinical_data['Age'], bins=20, edgecolor='black')
            axes[0, 1].set_title('Distribuição de Idade dos Pacientes')
            axes[0, 1].set_xlabel('Idade')
            axes[0, 1].set_ylabel('Frequência')
            
            # 3. Distribuição por receptor ER
            er_counts = clinical_data['ER'].value_counts()
            axes[1, 0].pie(er_counts.values, labels=['ER+', 'ER-'], autopct='%1.1f%%')
            axes[1, 0].set_title('Distribuição por Receptor ER')
            
            # 4. Tamanho do tumor por classe ALN
            for aln_class in ['N0', 'N+(1-2)', 'N+(>2)']:
                data = clinical_data[clinical_data['ALN status'] == aln_class]['Tumor size (cm)']
                axes[1, 1].hist(data, alpha=0.5, label=aln_class, bins=15)
            axes[1, 1].set_title('Tamanho do Tumor por Classe ALN')
            axes[1, 1].set_xlabel('Tamanho (cm)')
            axes[1, 1].set_ylabel('Frequência')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Salva a figura como imagem temporária
            temp_stats_path = Path(__file__).parent / "temp_statistics.png"
            plt.savefig(temp_stats_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Carrega e exibe a imagem salva
            stats_img = Image.open(temp_stats_path)
            photo = ImageTk.PhotoImage(stats_img)
            
            img_label = tk.Label(stats_frame, image=photo, bg='white')
            img_label.image = photo  # Mantém referência
            img_label.pack(fill=tk.BOTH, expand=True)
            
            # Remove o arquivo temporário
            if temp_stats_path.exists():
                os.remove(temp_stats_path)
            
            # Informações adicionais
            info_text = f"""
            Total de pacientes: {len(clinical_data)}
            Idade média: {clinical_data['Age'].mean():.1f} anos
            Tamanho médio do tumor: {clinical_data['Tumor size (cm)'].mean():.2f} cm
            """
            
            info_label = ttk.Label(parent,
                                 text=info_text,
                                 font=('Arial', 11))
            info_label.pack(pady=10)
            
        except Exception as e:
            error_label = ttk.Label(stats_frame,
                                  text=f"Erro ao gerar estatísticas:\n{str(e)}",
                                  font=('Arial', 12))
            error_label.pack(pady=50)


def main():
    """
    Função principal para executar a aplicação
    """
    root = tk.Tk()
    app = BreastCancerAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
