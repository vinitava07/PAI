import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Importa as classes de segmentação (assumindo que estão em módulos separados)
# from nucleus_segmentation import NucleusSegmentation
# from nucleus_segmentation_dl import DeepLearningNucleusSegmenter

class SegmentationInterface:
    """
    Interface gráfica para segmentação de núcleos
    Integra os métodos de segmentação com a aplicação principal
    """

    def __init__(self, parent_window):
        self.parent = parent_window
        self.current_image = None
        self.gray_image = None
        self.segmented_image = None
        self.nucleus_features = None
        self.statistics = None

        # Cria a janela de segmentação
        self.window = tk.Toplevel(parent_window)
        self.window.title("Segmentação de Núcleos")
        self.window.geometry("1200x800")

        self.setup_ui()

    def setup_ui(self):
        """Configura a interface de usuário"""

        # Frame principal
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configuração de pesos para redimensionamento
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # --- Controles superiores ---
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Botão para carregar imagem
        ttk.Button(control_frame, text="Carregar Imagem",
                   command=self.load_image).grid(row=0, column=0, padx=5)

        # Seleção do método de segmentação
        ttk.Label(control_frame, text="Método:").grid(row=0, column=1, padx=(20, 5))
        self.method_var = tk.StringVar(value="watershed")
        method_combo = ttk.Combobox(control_frame, textvariable=self.method_var,
                                   values=["watershed", "advanced", "deep_learning"],
                                   state="readonly", width=15)
        method_combo.grid(row=0, column=2, padx=5)

        # Botão para executar segmentação
        ttk.Button(control_frame, text="Segmentar Núcleos",
                   command=self.run_segmentation).grid(row=0, column=3, padx=5)

        # Botão para exportar resultados
        ttk.Button(control_frame, text="Exportar Resultados",
                   command=self.export_results).grid(row=0, column=4, padx=5)

        # --- Área de visualização ---
        # Frame para imagens
        image_frame = ttk.LabelFrame(main_frame, text="Visualização", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Canvas para exibir imagens
        self.canvas = tk.Canvas(image_frame, bg="gray", width=600, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Frame para controles de visualização
        view_control_frame = ttk.Frame(image_frame)
        view_control_frame.pack(fill=tk.X, pady=(5, 0))

        self.view_var = tk.StringVar(value="original")
        ttk.Radiobutton(view_control_frame, text="Original",
                        variable=self.view_var, value="original",
                        command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_control_frame, text="Tons de Cinza",
                        variable=self.view_var, value="gray",
                        command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_control_frame, text="Segmentada",
                        variable=self.view_var, value="segmented",
                        command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_control_frame, text="Contornos",
                        variable=self.view_var, value="contours",
                        command=self.update_view).pack(side=tk.LEFT, padx=5)

        # --- Área de estatísticas ---
        stats_frame = ttk.LabelFrame(main_frame, text="Estatísticas", padding="10")
        stats_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Notebook para diferentes visualizações
        self.notebook = ttk.Notebook(stats_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Aba de estatísticas textuais
        self.stats_text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_text_frame, text="Estatísticas")

        # Text widget para estatísticas
        self.stats_text = tk.Text(self.stats_text_frame, wrap=tk.WORD,
                                 width=40, height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar para o texto
        scrollbar = ttk.Scrollbar(self.stats_text_frame, orient=tk.VERTICAL,
                                 command=self.stats_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        # Aba para gráficos
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Gráficos")

        # --- Barra de progresso ---
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2,
                          sticky=(tk.W, tk.E), pady=(10, 0))

        # --- Barra de status ---
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2,
                       sticky=(tk.W, tk.E), pady=(5, 0))

    def load_image(self):
        """Carrega uma imagem para segmentação"""
        filename = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.tif *.bmp"),
                      ("Todos os arquivos", "*.*")]
        )

        if filename:
            try:
                # Carrega a imagem
                self.current_image = cv2.imread(filename)
                self.current_image_rgb = cv2.cvtColor(self.current_image,
                                                      cv2.COLOR_BGR2RGB)

                # Converte para tons de cinza
                self.gray_image = cv2.cvtColor(self.current_image_rgb,
                                              cv2.COLOR_RGB2GRAY)

                # Atualiza visualização
                self.update_view()
                self.status_var.set(f"Imagem carregada: {os.path.basename(filename)}")

                # Limpa resultados anteriores
                self.segmented_image = None
                self.nucleus_features = None
                self.stats_text.delete(1.0, tk.END)

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")

    def run_segmentation(self):
        """Executa a segmentação de núcleos"""
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        try:
            # Inicia progresso
            self.progress.start()
            self.status_var.set("Segmentando núcleos...")
            self.window.update()

            method = self.method_var.get()

            if method == "deep_learning":
                # Usa segmentação por deep learning
                self.segment_with_dl()
            else:
                # Usa métodos tradicionais
                self.segment_traditional(method)

            # Para progresso
            self.progress.stop()
            self.status_var.set("Segmentação concluída!")

            # Atualiza visualização
            self.view_var.set("segmented")
            self.update_view()

            # Exibe estatísticas
            self.display_statistics()

            # Cria gráficos
            self.create_plots()

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Erro", f"Erro na segmentação: {str(e)}")
            self.status_var.set("Erro na segmentação")

    def segment_traditional(self, method):
        """Segmentação usando métodos tradicionais"""
        # Aqui você integraria com a classe NucleusSegmentation
        # Por enquanto, vamos simular

        # Simulação de segmentação
        # Em produção, use: segmenter = NucleusSegmentation()
        # features, stats, labeled = segmenter.process_image(self.current_image, method)

        # Placeholder para demonstração
        h, w = self.gray_image.shape
        self.segmented_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Simula algumas regiões segmentadas
        num_nuclei = np.random.randint(50, 150)

        # Gera estatísticas simuladas
        self.statistics = {
            'num_nuclei': num_nuclei,
            'area': {
                'mean': np.random.uniform(100, 500),
                'std': np.random.uniform(50, 150)
            },
            'circularity': {
                'mean': np.random.uniform(0.6, 0.9),
                'std': np.random.uniform(0.05, 0.15)
            },
            'eccentricity': {
                'mean': np.random.uniform(0.3, 0.7),
                'std': np.random.uniform(0.1, 0.2)
            },
            'normalized_nn_distance': {
                'mean': np.random.uniform(2, 5),
                'std': np.random.uniform(0.5, 1.5)
            }
        }

        # Gera dados para planilha
        self.generate_feature_dataframe()

    def segment_with_dl(self):
        """Segmentação usando deep learning"""
        # Aqui você integraria com DeepLearningNucleusSegmenter
        # Por enquanto, vamos simular

        messagebox.showinfo("Info",
                           "Segmentação por Deep Learning requer modelo treinado.\n" +
                           "Usando simulação para demonstração.")

        # Simula segmentação por DL
        self.segment_traditional("advanced")

    def generate_feature_dataframe(self):
        """Gera DataFrame com características dos núcleos"""
        if self.statistics:
            num_nuclei = self.statistics['num_nuclei']

            # Gera dados simulados para cada núcleo
            data = {
                'nucleus_id': range(1, num_nuclei + 1),
                'area': np.random.normal(self.statistics['area']['mean'],
                                       self.statistics['area']['std'], num_nuclei),
                'circularity': np.random.normal(self.statistics['circularity']['mean'],
                                              self.statistics['circularity']['std'], num_nuclei),
                'eccentricity': np.random.normal(self.statistics['eccentricity']['mean'],
                                               self.statistics['eccentricity']['std'], num_nuclei),
                'nn_distance_normalized': np.random.normal(
                    self.statistics['normalized_nn_distance']['mean'],
                    self.statistics['normalized_nn_distance']['std'], num_nuclei)
            }

            self.nucleus_features = pd.DataFrame(data)

    def display_statistics(self):
        """Exibe estatísticas na interface"""
        if self.statistics:
            self.stats_text.delete(1.0, tk.END)

            text = f"ESTATÍSTICAS DA SEGMENTAÇÃO\n"
            text += f"{'='*40}\n\n"
            text += f"Número de núcleos detectados: {self.statistics['num_nuclei']}\n\n"

            text += f"CARACTERÍSTICAS MORFOMÉTRICAS\n"
            text += f"{'-'*40}\n\n"

            for feature, values in self.statistics.items():
                if feature != 'num_nuclei' and isinstance(values, dict):
                    text += f"{feature.replace('_', ' ').title()}:\n"
                    text += f"  Média: {values['mean']:.3f}\n"
                    text += f"  Desvio Padrão: {values['std']:.3f}\n\n"

            self.stats_text.insert(1.0, text)

    def create_plots(self):
        """Cria gráficos de dispersão das características"""
        if self.nucleus_features is None:
            return

        # Limpa frame de plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Cria figura
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle('Análise de Características dos Núcleos')

        # Plot 1: Área vs Circularidade
        axes[0, 0].scatter(self.nucleus_features['area'],
                          self.nucleus_features['circularity'],
                          alpha=0.6, color='blue')
        axes[0, 0].set_xlabel('Área')
        axes[0, 0].set_ylabel('Circularidade')
        axes[0, 0].set_title('Área vs Circularidade')

        # Plot 2: Área vs Excentricidade
        axes[0, 1].scatter(self.nucleus_features['area'],
                          self.nucleus_features['eccentricity'],
                          alpha=0.6, color='green')
        axes[0, 1].set_xlabel('Área')
        axes[0, 1].set_ylabel('Excentricidade')
        axes[0, 1].set_title('Área vs Excentricidade')

        # Plot 3: Histograma de áreas
        axes[1, 0].hist(self.nucleus_features['area'], bins=20,
                       color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Área')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição de Áreas')

        # Plot 4: Boxplot das características
        data_to_plot = [
            self.nucleus_features['area'] / self.nucleus_features['area'].max(),
            self.nucleus_features['circularity'],
            self.nucleus_features['eccentricity'],
            self.nucleus_features['nn_distance_normalized'] /
            self.nucleus_features['nn_distance_normalized'].max()
        ]

        axes[1, 1].boxplot(data_to_plot,
                          labels=['Área\n(norm)', 'Circular.', 'Excent.', 'Dist. NN\n(norm)'])
        axes[1, 1].set_ylabel('Valor Normalizado')
        axes[1, 1].set_title('Comparação de Características')

        plt.tight_layout()

        # Adiciona canvas ao frame
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_view(self):
        """Atualiza a visualização da imagem"""
        view_type = self.view_var.get()

        if view_type == "original" and self.current_image_rgb is not None:
            self.display_image(self.current_image_rgb)
        elif view_type == "gray" and self.gray_image is not None:
            # Converte para RGB para exibição
            gray_rgb = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2RGB)
            self.display_image(gray_rgb)
        elif view_type == "segmented" and self.segmented_image is not None:
            self.display_image(self.segmented_image)
        elif view_type == "contours" and self.segmented_image is not None:
            self.display_contours()

    def display_image(self, image):
        """Exibe uma imagem no canvas"""
        # Redimensiona imagem para caber no canvas
        h, w = image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calcula fator de escala
            scale = min(canvas_width/w, canvas_height/h, 1.0)
            new_width = int(w * scale)
            new_height = int(h * scale)

            # Redimensiona
            resized = cv2.resize(image, (new_width, new_height))

            # Converte para PIL Image
            pil_image = Image.fromarray(resized)
            self.photo = ImageTk.PhotoImage(pil_image)

            # Limpa canvas e exibe imagem
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2,
                                   image=self.photo, anchor=tk.CENTER)

    def display_contours(self):
        """Exibe contornos dos núcleos sobre a imagem original"""
        if self.current_image_rgb is not None and self.segmented_image is not None:
            # Cria cópia da imagem original
            contour_image = self.current_image_rgb.copy()

            # Simula desenho de contornos (em produção, use os contornos reais)
            # Por enquanto, vamos criar alguns contornos aleatórios
            overlay = contour_image.copy()

            # Adiciona alguns círculos simulando núcleos
            h, w = contour_image.shape[:2]
            num_circles = 50

            for _ in range(num_circles):
                center_x = np.random.randint(20, w-20)
                center_y = np.random.randint(20, h-20)
                radius = np.random.randint(10, 30)

                cv2.circle(overlay, (center_x, center_y), radius,
                          (0, 255, 0), 2)

            # Mistura com transparência
            alpha = 0.7
            contour_image = cv2.addWeighted(contour_image, alpha,
                                          overlay, 1-alpha, 0)

            self.display_image(contour_image)

    def export_results(self):
        """Exporta os resultados da segmentação"""
        if self.nucleus_features is None:
            messagebox.showwarning("Aviso",
                                 "Não há resultados para exportar!")
            return

        # Diálogo para salvar arquivo
        filename = filedialog.asksaveasfilename(
            title="Exportar resultados",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")]
        )

        if filename:
            try:
                if filename.endswith('.xlsx'):
                    # Exporta para Excel com múltiplas abas
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        # Dados dos núcleos
                        self.nucleus_features.to_excel(writer,
                                                     sheet_name='Núcleos',
                                                     index=False)

                        # Estatísticas resumidas
                        stats_df = pd.DataFrame([
                            {'Característica': 'Número de Núcleos',
                             'Valor': self.statistics['num_nuclei']},
                            {'Característica': 'Área Média',
                             'Valor': f"{self.statistics['area']['mean']:.2f} ± {self.statistics['area']['std']:.2f}"},
                            {'Característica': 'Circularidade Média',
                             'Valor': f"{self.statistics['circularity']['mean']:.3f} ± {self.statistics['circularity']['std']:.3f}"},
                            {'Característica': 'Excentricidade Média',
                             'Valor': f"{self.statistics['eccentricity']['mean']:.3f} ± {self.statistics['eccentricity']['std']:.3f}"},
                            {'Característica': 'Distância NN Normalizada Média',
                             'Valor': f"{self.statistics['normalized_nn_distance']['mean']:.3f} ± {self.statistics['normalized_nn_distance']['std']:.3f}"}
                        ])

                        stats_df.to_excel(writer,
                                        sheet_name='Estatísticas',
                                        index=False)

                else:  # CSV
                    self.nucleus_features.to_csv(filename, index=False)

                messagebox.showinfo("Sucesso",
                                  f"Resultados exportados para:\n{filename}")

            except Exception as e:
                messagebox.showerror("Erro",
                                   f"Erro ao exportar: {str(e)}")


# Função para integrar com a aplicação principal
def open_segmentation_window(parent):
    """
    Abre a janela de segmentação

    Args:
        parent: Janela pai da aplicação
    """
    segmentation_window = SegmentationInterface(parent)
    return segmentation_window


if __name__ == "__main__":
    # Teste standalone
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    seg_interface = SegmentationInterface(root)

    root.mainloop()
