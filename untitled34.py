import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

st.title('NDVI Image Processing')

# Permitir o upload de várias imagens TIFF
uploaded_files = []
while True:
    uploaded_file = st.file_uploader("Choose Sentinel-2 image:", type="tif")
    if uploaded_file is None:
        break
    uploaded_files.append(uploaded_file)

# Processar cada imagem TIFF
for uploaded_file in uploaded_files:
    # Carregar a imagem TIFF usando o OpenCV
    image = cv2.imread(uploaded_file)

    # ... (resto do c
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
    st.image('image.jpg', width=500)

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

