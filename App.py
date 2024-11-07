# Игнорирование уведомлений в консоли
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from tkinter import *
import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageOps import invert
from keras.src.saving import load_model


def draw(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    drawImg.ellipse((x1, y1, x2, y2), fill='black')


def clear():
    canvas.delete('all')
    drawImg.rectangle((0, 0, 300, 300), fill='white')
    answer.set('')


def recognize():
    model = load_model('myModel.keras')
    res = model.predict([normalize(image)])[0]
    answer.set(f'{list(res).index(np.max(res))}')


def normalize(img):
    img = invert(img.resize((28, 28)))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    return img


window = Tk()
window.title('Распознавание')
window.resizable(width=False, height=False)

window.columnconfigure(index=1, weight=1)
window.rowconfigure(index=2, weight=1)

canvas = Canvas(window, bg='white', width=300, height=300)
canvas.grid(columnspan=7, padx=10, pady=20)
canvas.bind('<B1-Motion>', draw)

image = Image.new('RGB', (300, 300), 'white')
drawImg = ImageDraw.Draw(image)

answer = StringVar()

Button(window, text='Распознать', font=("Cascadia Code SemiLight", "12"),
       width=13, command=recognize, pady=8).grid(row=1, padx=25)
Button(window, text='Очистить', font=("Cascadia Code SemiLight", "12"),
       width=13, command=clear, pady=8).grid(row=1, column=1, padx=20)
Label(window, text='Ответ: ', font=("Cascadia Code", "14"), pady=15).grid(row=2, column=0)
Entry(window, textvariable=answer, font=("Cascadia Code SemiLight", "14"),
      state=DISABLED, width=10).grid(row=2, column=1)

window.mainloop()
