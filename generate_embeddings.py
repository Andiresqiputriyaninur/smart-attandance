"""
Script untuk generate face embeddings dari dataset foto mahasiswa
Menggunakan model ArcFace untuk ekstraksi fitur wajah
"""

import os
import cv2
import pickle
import numpy as np
from app.face_recognition import FaceRecognition

def load_face_dataset(data_dir="data/faces"):
    """Load semua foto dari dataset dan label nama"""
    faces = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"❌ Direktori {data_dir} tidak ditemukan!")
        return faces, labels

    print(f"📁 Loading dataset dari: {data_dir}")

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"👤 Processing: {person_name}")

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Tidak dapat load: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces.append(image_rgb)
            labels.append(person_name)

            print(f"   ✅ {image_name}")

    print(f"📊 Total images loaded: {len(faces)}")
    return faces, labels

def generate_embeddings(faces, labels):
    """Generate embeddings menggunakan ArcFace model dengan quality filtering"""
    print("🧠 Initializing Face Recognition model...")

    try:
        face_recognizer = FaceRecognition()
        print("✅ Model ArcFace berhasil dimuat")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

    embeddings = []
    valid_labels = []

    print("🔄 Generating high-quality embeddings...")

    person_embeddings = {}

    for i, (face, label) in enumerate(zip(faces, labels)):
        try:
            embedding = face_recognizer.get_embedding(face)

            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)

                if label not in person_embeddings:
                    person_embeddings[label] = []
                person_embeddings[label].append(embedding)
                print(f"   ✅ {i+1}/{len(faces)}: {label}")
            else:
                print(f"   ❌ {i+1}/{len(faces)}: {label} - No face detected")

        except Exception as e:
            print(f"   ❌ {i+1}/{len(faces)}: {label} - Error: {e}")

    print("🔍 Quality filtering embeddings...")
    for person, embs in person_embeddings.items():
        if len(embs) > 4:
            mean_emb = np.mean(embs, axis=0)
            similarities = [np.dot(emb, mean_emb) for emb in embs]

            best_indices = np.argsort(similarities)[-4:]
            selected_embs = [embs[i] for i in best_indices]
            print(f"   📊 {person}: Selected 4 best from {len(embs)} embeddings")
        else:
            selected_embs = embs
            print(f"   📊 {person}: Using all {len(embs)} embeddings")

        for emb in selected_embs:
            embeddings.append(emb)
            valid_labels.append(person)

    print(f"📊 Generated {len(embeddings)} high-quality embeddings")
    return np.array(embeddings), valid_labels

def save_embeddings(embeddings, labels, output_path="models/embeddings.pkl"):
    """Simpan embeddings ke file pickle"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        face_database = {}
        for embedding, label in zip(embeddings, labels):
            if label not in face_database:
                face_database[label] = []
            face_database[label].append(embedding)

        with open(output_path, 'wb') as f:
            pickle.dump(face_database, f)

        print(f"💾 Embeddings saved to: {output_path}")
        print(f"📊 Total embeddings: {len(embeddings)}")
        print(f"👥 Unique persons: {len(face_database)}")

        return True

    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        return False

def main():
    """Fungsi utama untuk generate embeddings"""
    print("🎯 === FACE EMBEDDINGS GENERATOR ===")
    print()

    faces, labels = load_face_dataset()

    if len(faces) == 0:
        print("❌ Tidak ada foto yang ditemukan!")
        print("💡 Pastikan foto ada di folder 'data/faces/[nama_mahasiswa]/'")
        return

    embeddings, valid_labels = generate_embeddings(faces, labels)

    if embeddings is None or len(embeddings) == 0:
        print("❌ Gagal generate embeddings!")
        return

    if save_embeddings(embeddings, valid_labels):
        print("\n🎉 SUKSES! Face embeddings berhasil di-generate")
        print("💡 Sekarang sistem siap untuk pengenalan wajah")

        unique_persons = set(valid_labels)
        print(f"\n📈 STATISTIK:")
        for person in unique_persons:
            count = valid_labels.count(person)
            print(f"   👤 {person}: {count} embeddings")
    else:
        print("❌ Gagal menyimpan embeddings!")

if __name__ == "__main__":
    main()