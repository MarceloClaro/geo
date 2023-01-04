import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

st.title('NDVI Image Processing')

# Create a file uploader widget
uploaded_files = st.file_uploader("Choose Sentinel-2 images:", type="tif", multiple=True)

if uploaded_files is not None:
    # Display the uploaded images
    for uploaded_file in uploaded_files:
        # Load the image file into memory
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image)

        # Extract the metadata
        metadata = image.tag_v2

        # Display the metadata
        st.write(metadata)

    # Processar cada imagem TIFF
    for uploaded_file in uploaded_files:
        # Carregar a imagem TIFF usando o OpenCV
        image = cv2.imread(uploaded_file)

        # Converter a imagem para uma representação de valores de pixel de ponto flutuante
        image = image.astype(np.float32)

        # Obter as bandas de infravermelho e vermelho da imagem
        infrared_band = image[:,:,0]
        red_band = image[:,:,1]

        # Calcular o NDVI
        ndvi = (infrared_band - red_band) / (infrared_band + red_band)

        # Adicionar opções de personalização do processamento da imagem
        bands = st.sidebar.multiselect(
            'Choose bands for NDVI calculation:',
            ['Infrared', 'Red'],
            ['Infrared', 'Red']
        )
        if 'Infrared' in bands:
            infrared_band_index = 0
        else:
            infrared_band_index = st.sidebar.slider(
                'Infrared band index:',
                0, image.shape[2]-1, 0, 1
            )
        if 'Red' in bands:
            red_band_index = 1
        else:
                        red_band_index = st.sidebar.slider(
                'Red band index:',
                0, image.shape[2]-1, 1, 1
            )
        infrared_band = image[:,:,infrared_band_index]
        red_band = image[:,:,red_band_index]
        ndvi = (infrared_band - red_band) / (infrared_band + red_band)

        # Exibir o mapa de calor da imagem NDVI
        st.image(ndvi, cmap='RdYlGn', interpolation='nearest')

        # Exibir o histograma da imagem NDVI
        plt.hist(ndvi.flatten())
        st.pyplot()

        # Realizar o cluster K-means com K=5
        k = st.sidebar.slider(
            'Number of clusters:',
            2, 10, 5, 1
        )
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(ndvi.reshape(-1, 1))
        clusters = kmeans.predict(ndvi.reshape(-1, 1))

        # Obter a porcentagem de pixels em cada cluster
        cluster_percentages = []
        for i in range(k):
            cluster_percentages.append(np.sum(clusters == i) / len(clusters))

        # Exibir os resultados
        st.write('Porcentagem de pixels em cada cluster:')
        st.write(cluster_percentages)

        # Adicionar opção de salvar os resultados do processamento em um arquivo
        save_results = st.sidebar.checkbox(
            'Save results to file?'
        )
        if save_results:
            file_format = st.sidebar.radio(
                'Choose file format:',
                ['TIFF', 'CSV']
            )
            if file_format == 'TIFF':
                # Salvar o mapa de calor da imagem NDVI como um arquivo TIFF
                image = Image.fromarray(ndvi)
                image_bytes = image.tobytes()
                image = Image.open(image_bytes)
                image.save('ndvi.tiff')
                st.success('Saved NDVI image to TIFF file.')
            else:
                # Salvar os resultados do processamento em um arquivo CSV
                import pandas as pd
                df = pd.DataFrame({                    'cluster': clusters,
                    'percentage': cluster_percentages
                })
                df.to_csv('results.csv', index=False)
                st.success('Saved results to CSV file.')

