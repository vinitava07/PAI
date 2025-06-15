import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import os

class ImageViewer(tk.Tk):
    def __init__(self, base_path, min_folder=1, max_folder=1058):
        super().__init__()
        self.title("Visualizador de Imagens com Zoom")
        self.geometry("800x700")

        self.base_path = base_path
        self.min_folder = min_folder
        self.max_folder = max_folder
        self.current_folder_number = min_folder
        self.current_image_path = None
        self.original_image = None # Imagem original da Pillow
        self.tk_image = None       # Imagem para o Tkinter
        self.zoom_level = 1.0
        self.image_on_canvas = None

        # --- Controles ---
        controls_frame = ttk.Frame(self, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls_frame, text="Anterior", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        
        self.folder_number_var = tk.StringVar(value=str(self.current_folder_number))
        self.folder_entry = ttk.Spinbox(
            controls_frame,
            from_=self.min_folder,
            to=self.max_folder,
            textvariable=self.folder_number_var,
            width=5,
            command=self.go_to_image_from_spinbox # Usar command para atualizar ao mudar valor
        )
        self.folder_entry.pack(side=tk.LEFT, padx=5)
        self.folder_entry.bind("<Return>", self.go_to_image_from_entry) # Atualizar com Enter

        ttk.Button(controls_frame, text="Próximo", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Ir", command=self.go_to_image_from_entry).pack(side=tk.LEFT, padx=5)

        ttk.Button(controls_frame, text="Zoom +", command=self.zoom_in).pack(side=tk.RIGHT, padx=5)
        ttk.Button(controls_frame, text="Zoom -", command=self.zoom_out).pack(side=tk.RIGHT, padx=5)
        ttk.Button(controls_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.RIGHT, padx=5)


        # --- Canvas para Imagem ---
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Informações ---
        info_frame = ttk.Frame(self, padding="5")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(info_frame, text="Nenhuma imagem carregada.")
        self.status_label.pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(info_frame, text=f"Zoom: {self.zoom_level:.1f}x")
        self.zoom_label.pack(side=tk.RIGHT)

        # Carregar a primeira imagem
        self.load_and_display_image()

    def find_image_in_folder(self, folder_path):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        if not os.path.isdir(folder_path):
            return None
        for fname in sorted(os.listdir(folder_path)): # Ordenar para consistência
            if fname.lower().endswith(valid_extensions):
                return os.path.join(folder_path, fname)
        return None

    def load_and_display_image(self):
        folder_path = os.path.join(self.base_path, str(self.current_folder_number))
        self.current_image_path = self.find_image_in_folder(folder_path)
        self.original_image = None # Resetar antes de tentar carregar

        if self.current_image_path and os.path.exists(self.current_image_path):
            print(f"[DEBUG] Tentando carregar: {self.current_image_path}")
            try:
                temp_image = Image.open(self.current_image_path)
                temp_image.load() # Força o carregamento dos dados
                self.original_image = temp_image

                print(f"[DEBUG] Pillow carregou: Formato={self.original_image.format}, Tamanho={self.original_image.size}, Modo={self.original_image.mode}")

            except UnidentifiedImageError:
                messagebox.showerror("Erro de Imagem", f"Não foi possível identificar o formato da imagem (Pillow UnidentifiedImageError): {self.current_image_path}")
                print(f"[DEBUG ERRO] Pillow UnidentifiedImageError para {self.current_image_path}")
            except FileNotFoundError:
                messagebox.showerror("Erro de Arquivo", f"Arquivo não encontrado: {self.current_image_path}")
                print(f"[DEBUG ERRO] FileNotFoundError para {self.current_image_path}")
            except OSError as e:
                messagebox.showerror("Erro de Imagem (OSError)", f"Erro ao ler o arquivo de imagem (pode estar corrompido ou truncado): {self.current_image_path}\nDetalhes: {e}")
                print(f"[DEBUG ERRO] OSError para {self.current_image_path} - {e}")
            except Exception as e:
                messagebox.showerror("Erro ao Carregar Imagem", f"Ocorreu um erro inesperado ao carregar a imagem {self.current_image_path}\nTipo: {type(e).__name__}\nDetalhes: {e}")
                print(f"[DEBUG ERRO] Exceção genérica para {self.current_image_path} - {type(e).__name__}: {e}")
        else:
            if not self.current_image_path:
                print(f"[DEBUG] Nenhuma imagem encontrada na pasta: {folder_path} para o número {self.current_folder_number}")
            else:
                print(f"[DEBUG] Caminho da imagem {self.current_image_path} não existe mais.")

        if self.original_image:
            print("[DEBUG] self.original_image existe, chamando display_image()")
            self.display_image()
            self.status_label.config(text=f"Pasta: {self.current_folder_number} | Imagem: {os.path.basename(self.current_image_path)}")
        else:
            print("[DEBUG] self.original_image é None, limpando canvas.")
            self.clear_canvas()
            if self.current_image_path:
                 self.status_label.config(text=f"Falha ao carregar: {os.path.basename(self.current_image_path)}")
            else:
                 self.status_label.config(text=f"Nenhuma imagem na pasta: {self.current_folder_number}")

        self.update_zoom_label()
        self.folder_number_var.set(str(self.current_folder_number))

    def display_image(self):
        if not self.original_image:
            print("[DEBUG] display_image: self.original_image é None. Chamando clear_canvas_completo.")
            self.clear_canvas_completo() # Uma função que realmente limpa tudo incluindo self.tk_image
            return
        
        print(f"[DEBUG] display_image: Iniciando. Modo original: {self.original_image.mode}, Tamanho original: {self.original_image.size}")

        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        print(f"[DEBUG] display_image: Canvas dimensions: {canvas_width}x{canvas_height}")


        if canvas_width <= 1 or canvas_height <= 1:
            print("[DEBUG] display_image: Canvas não tem tamanho ainda. Tentando novamente em 100ms.")
            self.after(100, self.display_image)
            return

        img_w, img_h = self.original_image.size
        
        zoomed_w = int(img_w * self.zoom_level)
        zoomed_h = int(img_h * self.zoom_level)
        print(f"[DEBUG] display_image: Tamanho original={img_w}x{img_h}, Nível de Zoom={self.zoom_level:.2f}, Tamanho com zoom={zoomed_w}x{zoomed_h}")

        if zoomed_w <= 0 or zoomed_h <= 0:
            print(f"[DEBUG ERRO] display_image: Dimensões com zoom inválidas ({zoomed_w}x{zoomed_h}). Chamando clear_canvas_completo.")
            self.clear_canvas_completo()
            self.status_label.config(text=f"Imagem com dimensões inválidas após zoom: {zoomed_w}x{zoomed_h}")
            return

        new_tk_image = None # Variável local temporária
        try:
            display_img_resized = self.original_image.resize((zoomed_w, zoomed_h), Image.LANCZOS)
            print(f"[DEBUG] display_image: Imagem redimensionada para {display_img_resized.size}, modo {display_img_resized.mode}")
            
            display_img_rgb = display_img_resized.convert("RGB")
            print(f"[DEBUG] display_image: Imagem convertida para RGB, novo modo {display_img_rgb.mode}")
            
            new_tk_image = ImageTk.PhotoImage(display_img_rgb) # Atribui à variável local
            print(f"[DEBUG] display_image: ImageTk.PhotoImage criada localmente. Tamanho: {new_tk_image.width()}x{new_tk_image.height()}")

        except Exception as e:
            print(f"[DEBUG ERRO] display_image: Exceção durante redimensionamento/conversão/PhotoImage: {type(e).__name__}: {e}")
            messagebox.showerror("Erro de Exibição", f"Erro ao processar imagem para exibição:\n{type(e).__name__}: {e}")
            self.clear_canvas_completo()
            return

        # Se havia uma imagem anterior no canvas, delete apenas o item do canvas
        if self.image_on_canvas:
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = None 
            print("[DEBUG] display_image: Item antigo do canvas deletado.")

        # AGORA atribua a nova imagem tk à variável de instância para mantê-la viva
        self.tk_image = new_tk_image 

        x = (canvas_width - zoomed_w) / 2
        y = (canvas_height - zoomed_h) / 2
        print(f"[DEBUG] display_image: Desenhando NOVA imagem no canvas (com self.tk_image) em x={x:.1f}, y={y:.1f}")
        
        self.image_on_canvas = self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        # self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image) # Adicionar esta linha para segurança extra da referência
        print("[DEBUG] display_image: Nova imagem desenhada no canvas.")


    def clear_canvas_completo(self): # Usada quando realmente queremos limpar tudo
        print("[DEBUG] clear_canvas_completo: Limpando canvas e self.tk_image.")
        if self.image_on_canvas:
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = None
        self.tk_image = None # Anula a referência da imagem Tkinter

    def next_image(self):
        if self.current_folder_number < self.max_folder:
            self.current_folder_number += 1
            self.reset_zoom_level() # Reset zoom ao mudar de imagem
            self.load_and_display_image()
        else:
            messagebox.showinfo("Fim", "Você chegou à última pasta.")

    def prev_image(self):
        if self.current_folder_number > self.min_folder:
            self.current_folder_number -= 1
            self.reset_zoom_level() # Reset zoom ao mudar de imagem
            self.load_and_display_image()
        else:
            messagebox.showinfo("Início", "Você chegou à primeira pasta.")

    def go_to_image_from_entry(self, event=None): # event é para o bind do <Return>
        try:
            num = int(self.folder_number_var.get())
            if self.min_folder <= num <= self.max_folder:
                if self.current_folder_number != num:
                    self.current_folder_number = num
                    self.reset_zoom_level()
                    self.load_and_display_image()
            else:
                messagebox.showwarning("Número Inválido", f"Por favor, insira um número entre {self.min_folder} e {self.max_folder}.")
                self.folder_number_var.set(str(self.current_folder_number)) # Restaura o valor válido
        except ValueError:
            messagebox.showerror("Entrada Inválida", "Por favor, insira um número válido.")
            self.folder_number_var.set(str(self.current_folder_number)) # Restaura o valor válido

    def go_to_image_from_spinbox(self):
        # O command do Spinbox já muda o valor de self.folder_number_var
        # Então só precisamos chamar a lógica de atualização
        self.go_to_image_from_entry()

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.update_zoom_label()
        self.display_image()

    def zoom_out(self):
        if self.zoom_level * 0.8 > 0.05 : # Evitar zoom muito pequeno
            self.zoom_level *= 0.8
            self.update_zoom_label()
            self.display_image()

    def reset_zoom(self):
        self.reset_zoom_level()
        self.display_image()

    def reset_zoom_level(self):
        self.zoom_level = 1.0
        self.update_zoom_label()

    def update_zoom_label(self):
        self.zoom_label.config(text=f"Zoom: {self.zoom_level:.2f}x")