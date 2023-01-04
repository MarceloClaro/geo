import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

st.title("Monitoria da vegetação")

# Permitir o upload de uma imagem TIFF
uploaded_file = st.file_uploader("Escolha a imagem Sentinel-2:", type="tif")

# Verificar se o usuário escolheu uma imagem
if uploaded_file is not None:
    # Salvar o arquivo carregado em um lugar específico
    filename = "minha_imagem.tif"
    open(filename, "wb").write(uploaded_file.read())

    # Carregar a imagem TIFF usando o OpenCV e o caminho para o arquivo salvo
    image = cv2.imread(filename)

    # Converter a imagem para uma representação de valores de pixel de ponto flutuante
    image = image.astype(np.float32)
    # Obter as bandas de infravermelho e vermelho da imagem
    infrared_band = image[:,:,0]
    red_band = image[:,:,1]

    # Calcular o NDVI
    ndvi = (infrared_band - red_band) / (infrared_band + red_band)

    # Plotar o gráfico do NDVI
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar()
    st.pyplot()

    # Converter a imagem para JPEG
    image = Image.fromarray(image.astype(np.uint8))
    image_bytes = image.tobytes()
    image = Image.open(image_bytes)
    image.save('image.jpg')
    st.image('imagem.jpg', width=500)

    # Realizar o cluster K-means com K=5
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(ndvi.reshape(-1, 1))
    clusters = kmeans.predict(ndvi.reshape(-1, 1))

    # Obter a porcentagem de pixels em cada cluster
    cluster_percentages = []
    for i in range(5):
        cluster_percentages.append(np.sum(clusters == i) / len(clusters))

    # Exibir os resultados
    st.write('Porcentagem de pixels em cada cluster:')
    st.write(cluster_percentages)

    
