import streamlit as st
import pandas as pd
from lightfm import LightFM
from lightfm import cross_validation

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv('IMDb_Top_250_Cleaned.csv')
    return df

# Fungsi untuk melatih model
@st.cache_data
def train_model(df):
    # Memetakan film dan pengguna ke ID numerik
    movie_ids = {movie: idx for idx, movie in enumerate(df['Title'].unique())}
    user_ids = {user: idx for idx, user in enumerate(df['Age Rating'].unique())}

    # Mengonversi data ke dalam bentuk sparse matrix
    from scipy.sparse import coo_matrix

    rows = df['Age Rating'].map(user_ids.get)
    cols = df['Title'].map(movie_ids.get)
    data = df['IMDb Rating']

    matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    # Melatih model LightFM
    model = LightFM(loss='warp')
    model.fit(matrix, epochs=30, num_threads=2)

    return model, movie_ids, user_ids

# Fungsi untuk memberikan rekomendasi
def recommend(model, user_id, movie_ids, user_ids):
    user_idx = user_ids.get(user_id)
    if user_idx is None:
        return None
    scores = model.predict(user_idx, np.arange(len(movie_ids)))
    top_movies = movie_ids.keys()
    top_movie_ids = sorted(zip(scores, top_movies), reverse=True)[:5]
    return [movie for score, movie in top_movie_ids]

# Streamlit UI
st.title("IMDb Top 250 Movie Recommender")
st.write("Aplikasi ini memberikan rekomendasi berdasarkan age rating dan film yang Anda pilih.")

# Load data
df = load_data()

# Sidebar untuk memilih pengguna dan film
st.sidebar.header("Input Data")
user_id = st.sidebar.selectbox("Pilih Age Rating (User ID):", df["Age Rating"].unique())

# Train model
with st.spinner("Melatih model..."):
    model, movie_ids, user_ids = train_model(df)

st.success("Model telah dilatih!")

# Prediksi
if st.sidebar.button("Rekomendasi Film"):
    with st.spinner("Memproses rekomendasi..."):
        recommendations = recommend(model, user_id, movie_ids, user_ids)
    if recommendations:
        st.success(f"Rekomendasi untuk '{user_id}': {', '.join(recommendations)}")
    else:
        st.error("Pengguna atau film tidak ditemukan.")

# Tampilkan dataset
if st.checkbox("Tampilkan dataset"):
    st.write(df)
