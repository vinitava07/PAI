# Teste para verificar se o tkinter nativo funciona
import tkinter as tk
from PIL import Image, ImageTk

# Teste básico
root = tk.Tk()
root.withdraw()  # Esconde a janela

try:
    # Teste simples
    img = Image.new('RGB', (50, 50), 'red')
    photo = ImageTk.PhotoImage(img)
    print("✓ SUCCESS: tkinter nativo funciona com Pillow!")
except Exception as e:
    print(f"✗ ERROR: {e}")

root.destroy()
