import numpy as np
import os
import cv2
from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import matplotlib.pyplot as plt

import camera

global foto
global foto2
global foto3

global labelR, labelG, labelB

global opacidadeImg1, opacidadeImg2

global imgrgb,imgrgb2,imgrgb3





'''TODO
1 - Adicionar dois campos de texto (com valores globais) controlando a opacidade do método de mistura (mergeImage), alterando os valores de opacidade1 e opacidade2.
Lembrando de atualizar as variaveis imgrgb3 e foto3, e atualizar o canvas3

'''

class RectTracker:
	
	def __init__(self, canvas):
		self.canvas = canvas
		self.item = None
		
	def draw(self, start, end, **opts):
		"""Desenha retangulo"""
		return self.canvas.create_rectangle(*(list(start)+list(end)), **opts)
		
	def autodraw(self, **opts):
		"""Redesenha retangulo"""
		self.start = None
		self.canvas.bind("<Button-1>", self.__update, '+')
		self.canvas.bind("<B1-Motion>", self.__update, '+')
		self.canvas.bind("<ButtonRelease-1>", self.__stop, '+')
		
		self._command = opts.pop('command', lambda *args: None)
		self.rectopts = opts
		
	def __update(self, event):
		if not self.start:
			self.start = [event.x, event.y]
			return
		
		if self.item is not None:
			self.canvas.delete(self.item)
		self.item = self.draw(self.start, (event.x, event.y), **self.rectopts)
		self._command(self.start, (event.x, event.y))
		
	def __stop(self, event):
		self.start = None
		self.canvas.delete(self.item)
		self.item = None

# ------------------------------------------------- Funções Basicas -------------------------------------------------

def mouseRGB(event):
    global labelR, labelG, labelB
    x=event.y
    y=event.y
    colorsR = imgrgb[y,x,0]
    colorsG = imgrgb[y,x,1]
    colorsB = imgrgb[y,x,2]
    colors = imgrgb[y,x]
    labelR.configure(text=f'R: {colorsR}')
    labelG.configure(text=f'G: {colorsG}')
    labelB.configure(text=f'B: {colorsB}')

def setarOpacidade(op1,op2):
    global opacidadeImg1, opacidadeImg2
    opacidadeImg1, opacidadeImg2 = int(op1), int(op2)

def openImage1(canv):
    global imgrgb
    path = filedialog.askopenfilename()
    img = cv2.imread(path, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    global foto
    foto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb))
    canv.config(background="white")
    canv.create_image(0,0,image=foto, anchor=NW)

def openImage2(canv2):
    global imgrgb2
    path = filedialog.askopenfilename()
    img = cv2.imread(path, 1)
    imgrgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    global foto2
    foto2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb2))
    canv2.config(background="white")
    canv2.create_image(0,0,image=foto2, anchor=NW)

def salvar():
    global imgrgb3
    arquivo = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=( ("PNG file", "*.png"),("JPG file", "*.jpg"),("All Files", "*.*") ))
    output = PIL.Image.fromarray(imgrgb3)
    if arquivo:
        abs_path = os.path.abspath(arquivo.name)
        output.save(abs_path)

# ------------------------------------------------- Funções de Manip. de Imagem -------------------------------------------------

def mergeImage(canv3):
    # Ambas imagens tem que ter o mesmo tamanho!
    # A soma das duas opacidades tem q dar 100, ou 1 dps de multiplicar por 0.01
    global imgrgb,imgrgb2,imgrgb3
    global foto,foto2
    global opacidadeImg1,opacidadeImg2
    imgrgb3 = cv2.addWeighted(imgrgb, (opacidadeImg1*0.01), imgrgb2, (opacidadeImg2*0.01), 0)

    # Atualiza canvas 3
    global foto3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)

def escaladeCinza(canv3, imagem):

    global image
    global imgrgb3
    global foto3
    imagec = imagem.copy()
    w,h = imagem.shape[0],imagem.shape[1]
    for i in range (w):
        for j in range (h):
            cor = imagec[i][j]
            corMedia = (int(cor[0]) + int(cor[1]) + int(cor[2]))/3
            novaCor = (corMedia,corMedia,corMedia)
            imagec[i][j] = novaCor
    imgrgb3 = imagec

    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)
    return imgrgb3

limiar = 127
def limiarizacao(canv3, image):
    global imgrgb3
    global foto3
    escaladeCinza(canv3, image)
    w,h = imgrgb3.shape[0],imgrgb3.shape[1]
    for i in range (w):
        for j in range (h):
            cor = imgrgb3[i][j]
            if (cor[0] >= limiar):
                novaCor = (255,255,255) #Branco
            else:
                novaCor = (0,0,0) #Preto
            imgrgb3[i][j] = novaCor

    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)

def escalaNegativa(canv3 ,image):
    global imgrgb3
    global foto3
    imagec = image.copy()
    w,h = imagec.shape[0],imagec.shape[1]
    for i in range (w):
        for j in range (h):
            cor = imagec[i][j]
            novaCor = (255 - cor[0], 255 - cor[1], 255 - cor[2])
            imagec[i][j] = novaCor
    imgrgb3 = imagec
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)

def removeRuido(canv3,image,metodo):
    global imgrgb3
    global foto3
    imagec = image.copy()
    # Método: 0 - X
    # Método: 1 - CROSS +
    w,h = imagec.shape[0],imagec.shape[1]
    for i in range (1,w-1):
        for j in range (1,h-1):
            cor = imagec[i][j]
            vizinhosR = retornaVizinhosR(imagec, i, j,metodo)
            vizinhosR.append(cor[0])
            vizinhosG = retornaVizinhosG(imagec, i, j,metodo)
            vizinhosG.append(cor[1])
            vizinhosB = retornaVizinhosB(imagec, i, j,metodo)
            vizinhosB.append(cor[2])
            vizinhosR.sort()
            vizinhosG.sort()
            vizinhosB.sort()
            newR = mediana(vizinhosR)
            newG = mediana(vizinhosG)
            newB = mediana(vizinhosB)
            imagec[i][j] = (newR, newG, newB)
    imgrgb3 = imagec
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)

def mediana(lista):
    return lista[int(len(lista)/2)]

def retornaVizinhosR(image, x, y,metodo):
    if (metodo==0):
        vizinhosR = [int(image[x + 1][y - 1][0]), int(image[x + 1][y + 1][0]), int(image[x - 1][y + 1][0]), int(image[x - 1][y - 1][0])]
    else:
        vizinhosR = [int(image[x][y - 1][0]), int(image[x + 1][y][0]), int(image[x][y + 1][0]), int(image[x - 1][y][0])]
    return vizinhosR

def retornaVizinhosG(image, x, y,metodo):
    if (metodo==0):
        vizinhosG = [int(image[x + 1][y - 1][1]), int(image[x + 1][y + 1][1]), int(image[x - 1][y + 1][1]), int(image[x - 1][y - 1][1])]
    else:
        vizinhosG = [int(image[x][y - 1][1]), int(image[x + 1][y][1]), int(image[x][y + 1][1]), int(image[x - 1][y][1])]
    return vizinhosG

def retornaVizinhosB(image, x, y,metodo):
    if (metodo==0):
        vizinhosB = [int(image[x + 1][y - 1][2]), int(image[x + 1][y + 1][2]), int(image[x - 1][y + 1][2]), int(image[x - 1][y - 1][2])]
    else:
        vizinhosB = [int(image[x][y - 1][2]), int(image[x + 1][y][2]), int(image[x][y + 1][2]), int(image[x - 1][y][2])]
    return vizinhosB

def questao1(setores, canv3):
    global imgrgb
    global foto3
    imagemNova = imgrgb.copy()
    height, width = imagemNova.shape[:2]

    if (height % 2 == 1):
        if (width % 2 == 1):
            imagemNova = resize(imagemNova, width - 1, height - 1)
        else:
            imagemNova = resize(imagemNova, width, height - 1)
    else:
        if (width % 2 == 1):
            imagemNova = resize(imagemNova, width - 1, height)

    height, width = imagemNova.shape[:2]

    center = (int(height / 2), int(width / 2))

    matriz1 = imagemNova[0:center[0], 0:center[1]].copy()
    matriz2 = imagemNova[0:center[0], 0:center[1]].copy()
    matriz3 = imagemNova[0:center[0], 0:center[1]].copy()
    matriz4 = imagemNova[0:center[0], 0:center[1]].copy()

    for i in range(center[0]):
        for j in range(center[1]):
            matriz1[i][j] = imagemNova[i][j]

    for i in range(center[0]):
        for j in range(center[1], width):
            v = j - center[1]
            matriz2[i][v] = imagemNova[i][j]

    for i in range(center[0], height):
        for j in range(center[1]):
            u = i - center[0]
            matriz3[u][j] = imagemNova[i][j]

    for i in range(center[0], height):
        for j in range(center[1], width):
            u, v = i - center[0], j - center[1]
            matriz4[u][v] = imagemNova[i][j]

    if (setores[0] == 1):
        matriz1 = rotate(matriz1)

    if (setores[0] == 2):
        matriz2 = rotate(matriz2)

    if (setores[0] == 3):
        matriz3 = rotate(matriz3)

    if (setores[0] == 4):
        matriz4 = rotate(matriz4)

    if (setores[1] == 1):
        matriz1 = rotate(matriz1)

    if (setores[1] == 2):
        matriz2 = rotate(matriz2)

    if (setores[1] == 3):
        matriz3 = rotate(matriz3)

    if (setores[1] == 4):
        matriz4 = rotate(matriz4)

    cima = np.concatenate((matriz1, matriz2), axis=1)
    baixo = np.concatenate((matriz3, matriz4), axis=1)
    final = np.concatenate((cima, baixo), axis=0)

    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(final))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)

def rotate(imagem):
    img = imagem.copy()
    height, width = img.shape[:2]
    center1 = (int(width / 2), int(height / 2))
    matrizCentro = cv2.getRotationMatrix2D(center1, 180, 1)
    rotated180 = cv2.warpAffine(img, matrizCentro, (width, height))
    return rotated180

def resize(imagem,w,h):
    return cv2.resize(imagem, (w,h), interpolation = cv2.INTER_AREA)


def questao2(canv3):
    global imgrgb
    global foto3

    imagemNova = imgrgb.copy()
    heightOld, widthOld = imagemNova.shape[:2]

    if (heightOld > widthOld):
        imagemNova = resize(imagemNova, heightOld, heightOld)
    else:
        imagemNova = resize(imagemNova, widthOld, widthOld)

    R, G, B = cv2.split(imagemNova)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    equ = cv2.merge((output1_R, output1_G, output1_B))
    height, width = imagemNova.shape[:2]

    for i in range(height):
        for j in range(width):
            if i == j:
                imagemNova[i][j] = (0, 0, 0)
            if j > i:
                imagemNova[i][j] = equ[i][j]

    imagemNova = resize(imagemNova, widthOld, heightOld)

    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imagemNova))
    canv3.config(background="white")
    canv3.create_image(0, 0, image=foto3, anchor=NW)


def questao3 (lbResultado):
    global imgrgb
    global foto3
    imagemNova = imgrgb.copy()
    height, width = imagemNova.shape[:2]
    pontoInicial = [0,0]
    for i in range(height):
        loop = True
        for j in range(width):
            if preto(imagemNova[i][j]):
                pontoInicial = [i,j]
                loop = False
                break
        if not loop:
            break
    resultado = confereadjacente(imagemNova, pontoInicial[0], pontoInicial[1], pontoInicial, [0,0], True)
    lbResultado.configure(text=f'Resultado : {resultado}')

def confereadjacente(imagemNova,i,j, pontoInicial, pontoAnterior, primeira):
    if pontoInicial[0] == i and pontoInicial[1] == j and not primeira:
        return "FECHADO"
    elif preto(imagemNova[i + 1][j]) and not pontoAnterior == [i+1,j]:
        return confereadjacente(imagemNova, i+1,j,pontoInicial, [i,j], False)
    elif preto(imagemNova[i - 1][j]) and not pontoAnterior == [i-1,j]:
        return confereadjacente(imagemNova, i-1,j,pontoInicial, [i,j], False)
    elif preto(imagemNova[i][j + 1]) and not pontoAnterior == [i,j+1]:
        return confereadjacente(imagemNova, i,j+1,pontoInicial, [i,j], False)
    elif preto(imagemNova[i][j - 1])and not pontoAnterior == [i,j-1]:
        return confereadjacente(imagemNova, i,j-1,pontoInicial, [i,j], False)
    else:
        return "ABERTO"

def preto(cor):
    return cor[0] == 0 and cor[1] == 0 and cor[2] == 0

def branco(cor):
    return cor[0] == 255 and cor[1] == 255 and cor[2] == 255

def canny(valores, canv3, image):
    global foto3
    img = image.copy()
    img = escaladeCinza(canv3, img)
    w,h = img.shape[0],img.shape[1]
    new = cv2.Canny(img,valores[0],valores[1]) #OBS: Valores 100 e 200 São sliders que devem ser modificados para filtrar as bordas
    for i in range (w):
        for j in range (h):
            img[i][j] = new[i][j]
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)
    return img


def sobel(canv3, image):
    global foto3
    img = image.copy()
    img = escaladeCinza(canv3, img)
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    w,h = img.shape[0],img.shape[1]
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=1) # O valor do K determina o tamanho das bordas
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=1) # Somente usar K ímpar
    img_sobel = img_sobelx + img_sobely
    for i in range (w):
        for j in range (h):
            img[i][j] = img_sobel[i][j]
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)
    return img

def prewitt(canv3, image):
    global foto3
    img = image.copy()
    img = escaladeCinza(canv3, img)
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    w,h = image.shape[0],image.shape[1]
    kernelx = np.array([ [1,1,1], [0,0,0], [-1,-1,-1] ])
    kernely = np.array([ [-1,0,1], [-1,0,1], [-1,0,1] ])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    for i in range (w):
        for j in range (h):
            img[i][j] = img_prewitt[i][j]
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canv3.config(background="white")
    canv3.create_image(0,0,image=foto3, anchor=NW)
    return img

def bordas(canv3, imagem, imagem2):
    global foto3
    global imgrgb3
    img1 = imagem.copy()
    img2 = imagem2.copy()
    img3 = imagem2.copy()
    w,h = img1.shape[0],img1.shape[1]
    for i in range (w):
        for j in range (h):
            if (int(img1[i][j][0]) > 10):
                img3[i][j] = (0,255,0)
            else:
                img3[i][j] = img2[i][j]
    imgrgb3 = img3
    # Atualiza canvas 3
    foto3 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imgrgb3))
    canv3.config(background="white")
    canv3.create_image(0, 0, image=foto3, anchor=NW)

# ------------------------------------------------- MAIN -------------------------------------------------

def main():
    global labelR, labelG, labelB
    from random import shuffle
    janela = Tk()
    janela.title("PDI")
    janela.resizable(False,False)
    janela.geometry("1355x900")

    # ------ Frames --------------------------------------

    #Frame RGB
    frameRGB = LabelFrame(janela, text="Cores")
    frameRGB.place(x=5,y=5,height=100,width=280)

    #Frame Superior
    frameFuncoes = LabelFrame(janela, text="Funções")
    frameFuncoes.place(x=290,y=5,height=100,width=1060)

    #Frame Esquerdo
    frameManip = LabelFrame(janela, text="Manipulação de Imagem")
    frameManip.place(x=5,y=105,height=550,width=280)

    #Frame Esquerdo Inferior
    frameMerge = LabelFrame(janela, text="Mistura de Imagem")
    frameMerge.place(x=5,y=655,height=240,width=280)

    #Frame IMG1
    frameCanvas = LabelFrame(janela, text="Imagem 1")
    frameCanvas.place(x=290,y=105,height=690,width=350)

    #Frame IMG2
    frameCanvas2 = LabelFrame(janela, text="Imagem 2")
    frameCanvas2.place(x=645,y=105,height=690,width=350)

    #Frame IMG3
    frameCanvas3 = LabelFrame(janela, text="Imagem 3")
    frameCanvas3.place(x=1000,y=105,height=690,width=350)

    # ------ Canvases -----------------------------------

    #Canvas 1
    canv = Canvas(frameCanvas, width=480, height=690)
    #canv.config(background="white")
    canv.pack(fill=NONE, expand=YES)
    canv.bind("<Button-1>", mouseRGB)

    #Canvas 2
    canv2 = Canvas(frameCanvas2, width=480, height=690)
    #canv.config(background="white")
    canv2.pack(fill=NONE, expand=YES)
    canv2.bind("<Button-1>", mouseRGB)

    #Canvas 3
    canv3 = Canvas(frameCanvas3, width=480, height=690)
    #canv.config(background="white")
    canv3.pack(fill=NONE, expand=YES)
    canv3.bind("<Button-1>", mouseRGB)

    # ------ Retangulo de Seleção -----------------------------------

    #Desenha Retangulo Canvas 1
    rect = RectTracker(canv)
    def onDrag(start, end):
        global x,y
    rect.autodraw(fill="", width=2, command=onDrag)

    #Desenha Retangulo Canvas 2
    rect = RectTracker(canv2)
    def onDrag(start, end):
        global x,y
    rect.autodraw(fill="", width=2, command=onDrag)

    #Desenha Retangulo Canvas 3
    rect = RectTracker(canv3)
    def onDrag(start, end):
        global x,y
    rect.autodraw(fill="", width=2, command=onDrag)

    # ------ Criação de Botões -----------------------------------

    btImg1 = Button(frameFuncoes, width=45, text="Abrir Imagem 1", command=lambda: openImage1(canv))
    btImg1.place(x=10, y=15)

    btImg2 = Button(frameFuncoes, width=45, text="Abrir Imagem 2", command=lambda: openImage2(canv2))
    btImg2.place(x=365, y=15)

    btSalvar = Button(frameFuncoes, width=45, text="Salvar imagem editada", command=salvar)
    btSalvar.place(x=720, y=15)



# --------------------------------

    btMerge = Button(frameMerge, width=35, text="Misturar (imagens com mesmo tam.)", command=lambda: mergeImage(canv3))
    btMerge.place(x=10, y=85)

    btSalvarOpacidade = Button(frameMerge, width=5, height=3, text="ok ✔", command=lambda: setarOpacidade(enOpacidade1.get(), enOpacidade2.get()))
    btSalvarOpacidade.place(x=215, y=20)

    btBorda = Button(frameMerge, width=35, height=1, text="Misturar Borda", command=lambda: bordas(canv3, imgrgb, imgrgb2))
    btBorda.place(x=10, y=120)

    btGrayscale = Button(frameManip, width=35, text="Converter para Escala de Cinza",
                         command=lambda: escaladeCinza(canv3, imgrgb))
    btGrayscale.place(x=10, y=15)

    btLimiarizacao = Button(frameManip, width=35, text="Limirização", command=lambda: limiarizacao(canv3, imgrgb))
    btLimiarizacao.place(x=10, y=45)

    btEscalaNegativa = Button(frameManip, width=35, text="Converter para Escala Negativa",
                            command=lambda: escalaNegativa(canv3, imgrgb))
    btEscalaNegativa.place(x=10, y=75)

    btRemoveRuidox = Button(frameManip, width=35, text="Remover Ruído X",
                            command=lambda: removeRuido(canv3, imgrgb, 0))
    btRemoveRuidox.place(x=10, y=105)

    btRemoveRuidoy = Button(frameManip, width=35, text="Remover Ruído +",
                            command=lambda: removeRuido(canv3, imgrgb, 1))
    btRemoveRuidoy.place(x=10, y=135)

    btdeteccaofacial = Button(frameManip, width=35, text="Detecção Facial",
                            command=lambda: camera.iniciarcamera())
    btdeteccaofacial.place(x=10, y=165)

    btquestao1 = Button(frameManip, width=35, text="Questão 1",
                            command=lambda: questao1((int(enSetor1.get()), int(enSetor2.get())), canv3))
    btquestao1.place(x=10, y=195)

    btquestao2 = Button(frameManip, width=35, text="Questão 2",
                            command=lambda: questao2(canv3))
    btquestao2.place(x=10, y=225)

    btquestao3 = Button(frameManip, width=35, text="Questão 3",
                            command=lambda: questao3(lbresultado))
    btquestao3.place(x=10, y=255)

    btcanny = Button(frameManip, width=35, text="Canny",
                            command=lambda: canny((int(enCanny1.get()), int(enCanny2.get())), canv3, imgrgb))
    btcanny.place(x=10, y=370)

    btsobel = Button(frameManip, width=35, text="Sobel",
                            command=lambda: sobel(canv3, imgrgb))
    btsobel.place(x=10, y=465)

    btprewitt = Button(frameManip, width=35, text="Prewitt",
                            command=lambda: prewitt(canv3, imgrgb))
    btprewitt.place(x=10, y=495)


    # ------ Criação de Campos -----------------------------------

    enOpacidade1, enOpacidade2 = Entry(frameMerge, width=15), Entry(frameMerge, width=15)
    enOpacidade1.place(x=110, y=20), enOpacidade2.place(x=110, y=55)
    enSetor1, enSetor2 = Entry(frameManip, width=15), Entry(frameManip, width=15)
    enSetor1.place(x=10, y=320), enSetor2.place(x=140, y=320)

    enCanny1, enCanny2 = Entry(frameManip, width=15), Entry(frameManip, width=15)
    enCanny1.insert(0,"100"), enCanny2.insert(0,"200")
    enCanny1.place(x=10, y=425), enCanny2.place(x=140, y=425)

    # ------ Criação de Labels -----------------------------------

    lbOpacidade1, lbOpacidade2 = Label(frameMerge, text="Opacidade IMG1"), Label(frameMerge, text="Opacidade IMG2")
    lbOpacidade1.place(x=10, y=20), lbOpacidade2.place(x=10, y=55)
    lbSetor1, lbSetor2 = Label(frameManip, text="Setor 1"), Label(frameManip, text="Setor2")
    lbSetor1.place(x=10, y=295), lbSetor2.place(x=140, y=295)

    lbCanny1, lbCanny2 = Label(frameManip, text="Canny 1"), Label(frameManip, text="Canny 2")
    lbCanny1.place(x=10, y=400), lbCanny2.place(x=140, y=400)


    labelR = Label(frameRGB, text="R: ", fg="red")
    labelR.place(x=5, y=5)
    labelG = Label(frameRGB, text="G: ", fg="green")
    labelG.place(x=55, y=5)
    labelB = Label(frameRGB, text="B: ", fg="blue")
    labelB.place(x=105, y=5)
    lbresultado = Label(frameManip, text="Resultado: ")
    lbresultado.place(x=10, y=340)


    janela.mainloop()

main()