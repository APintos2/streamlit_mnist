import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../best_model.keras')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')

model = load_model('../best_model.keras')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('Reconocedor de números')
st.markdown('''
Escribe un número del 0 al 9!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 192
mode = st.checkbox("Modo Dibujo", True)
st.markdown('''
Desmarca el modo dibujo y haz doble click sobre el elemento que quieras
             eliminar para borrarlo.
''')

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')
    

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input del modelo:')
    st.image(rescaled)

if st.button('Predecir'):
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_test = x_test.astype("float32")/255
    val = model.predict(x_test.reshape(-1, 28, 28, 1))
    st.write(f'Resultado: {np.argmax(val[0])}')
    st.bar_chart(val[0])