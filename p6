# Mengimport library numpy dan pandas
import numpy as np
import pandas as pd

# Mengimport dataset
dataset = pd.read_csv('dataset.csv')

# Membagi dataset menjadi data training dan data testing
X_train = dataset.iloc[:80, :-1].values # Mengambil 80 baris pertama sebagai data training
y_train = dataset.iloc[:80, -1].values # Mengambil kelas dari 80 baris pertama sebagai label training
X_test = dataset.iloc[80:, :-1].values # Mengambil 20 baris terakhir sebagai data testing
y_test = dataset.iloc[80:, -1].values # Mengambil kelas dari 20 baris terakhir sebagai label testing

# Mendefinisikan fungsi untuk menghitung jarak euclidean
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

# Mendefinisikan fungsi untuk menentukan k tetangga terdekat
def get_k_neighbors(X_train, y_train, x_test, k):
  distances = [] # Membuat list kosong untuk menyimpan jarak
  for i in range(len(X_train)): # Melakukan iterasi untuk setiap data training
    distance = euclidean_distance(X_train[i], x_test) # Menghitung jarak antara data training dan data testing
    distances.append((distance, y_train[i])) # Menambahkan jarak dan label ke list distances
  distances.sort() # Mengurutkan list distances berdasarkan jarak terkecil
  neighbors = [] # Membuat list kosong untuk menyimpan tetangga
  for i in range(k): # Melakukan iterasi sebanyak k kali
    neighbors.append(distances[i][1]) # Menambahkan label dari k tetangga terdekat ke list neighbors
  return neighbors # Mengembalikan list neighbors

# Mendefinisikan fungsi untuk melakukan voting kelas
def vote(neighbors):
  counts = {} # Membuat dictionary kosong untuk menyimpan jumlah kelas
  for neighbor in neighbors: # Melakukan iterasi untuk setiap tetangga
    if neighbor not in counts: # Jika kelas tetangga belum ada di dictionary counts
      counts[neighbor] = 1 # Membuat key baru dengan value 1
    else: # Jika kelas tetangga sudah ada di dictionary counts
      counts[neighbor] += 1 # Menambahkan value dengan 1
  max_count = max(counts.values()) # Mencari nilai maksimum dari dictionary counts
  for key, value in counts.items(): # Melakukan iterasi untuk setiap pasangan key dan value di dictionary counts
    if value == max_count: # Jika value sama dengan nilai maksimum
      return key # Mengembalikan key sebagai kelas mayoritas

# Menerapkan fungsi-fungsi pada data testing dan membandingkan hasilnya dengan kelas sebenarnya
y_pred = [] # Membuat list kosong untuk menyimpan prediksi kelas
k = 5 # Menentukan nilai k
for x_test in X_test: # Melakukan iterasi untuk setiap data testing
  neighbors = get_k_neighbors(X_train, y_train, x_test, k) # Mendapatkan k tetangga terdekat dari data testing
  prediction = vote(neighbors) # Melakukan voting kelas dari k tetangga terdekat
  y_pred.append(prediction) # Menambahkan prediksi ke list y_pred

# Menghitung akurasi dari algoritma K-NN dengan menggunakan confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score 
cm = confusion_matrix(y_test, y_pred) # Membuat confusion matrix dari label sebenarnya dan prediksi label 
print(cm) 
accuracy = accuracy_score(y_test, y_pred) # Menghitung akurasi dari label sebenarnya dan prediksi label 
print(accuracy)
