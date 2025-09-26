"""
Script untuk regenerate face embeddings
Digunakan ketika ada perubahan dataset atau ingin refresh model
"""

import os
import pickle
import shutil
from datetime import datetime
from generate_embeddings import main as generate_main

def backup_existing_embeddings():
    """Backup file embeddings yang sudah ada"""
    embeddings_path = "models/embeddings.pkl"

    if os.path.exists(embeddings_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"models/embeddings_backup_{timestamp}.pkl"

        try:
            shutil.copy2(embeddings_path, backup_path)
            print(f"💾 Backup embeddings lama: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"⚠️ Gagal backup: {e}")
            return None
    else:
        print("ℹ️ Tidak ada embeddings lama untuk di-backup")
        return None

def check_dataset_info():
    """Cek informasi dataset yang ada"""
    faces_dir = "data/faces"

    if not os.path.exists(faces_dir):
        print("❌ Folder dataset tidak ditemukan!")
        return False

    students = []
    total_photos = 0

    for student_name in os.listdir(faces_dir):
        student_path = os.path.join(faces_dir, student_name)

        if not os.path.isdir(student_path):
            continue

        photos = [f for f in os.listdir(student_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        students.append((student_name, len(photos)))
        total_photos += len(photos)

    print(f"📊 INFORMASI DATASET:")
    print(f"   👥 Total mahasiswa: {len(students)}")
    print(f"   📷 Total foto: {total_photos}")
    print()

    print("📋 Detail per mahasiswa:")
    for name, count in sorted(students):
        status = "✅" if count >= 3 else "⚠️"
        print(f"   {status} {name}: {count} foto")

    insufficient = [name for name, count in students if count < 3]
    if insufficient:
        print(f"\n⚠️ Mahasiswa dengan foto kurang dari 3:")
        for name in insufficient:
            print(f"   - {name}")
        print("💡 Disarankan minimal 3-6 foto per mahasiswa untuk akurasi optimal")

    return len(students) > 0

def check_existing_embeddings():
    """Cek informasi embeddings yang sudah ada"""
    embeddings_path = "models/embeddings.pkl"

    if not os.path.exists(embeddings_path):
        print("ℹ️ Belum ada file embeddings")
        return

    try:
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)

        embeddings = data.get('embeddings', [])
        labels = data.get('labels', [])

        print(f"📄 EMBEDDINGS YANG ADA:")
        print(f"   📊 Total embeddings: {len(embeddings)}")
        print(f"   👥 Unique persons: {len(set(labels))}")

        from collections import Counter
        label_counts = Counter(labels)

        print("\n📋 Embeddings per mahasiswa:")
        for name, count in sorted(label_counts.items()):
            print(f"   - {name}: {count} embeddings")

        file_size = os.path.getsize(embeddings_path) / 1024 / 1024
        file_time = datetime.fromtimestamp(os.path.getmtime(embeddings_path))
        print(f"\n📁 File info:")
        print(f"   📏 Size: {file_size:.2f} MB")
        print(f"   🕒 Last modified: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ Error reading embeddings: {e}")

def main():
    """Fungsi utama regenerate embeddings"""
    print("🔄 === REGENERATE FACE EMBEDDINGS ===")
    print()

    if not check_dataset_info():
        print("❌ Tidak ada dataset untuk diproses!")
        return

    print()

    check_existing_embeddings()

    print()

    print("⚠️ PERINGATAN:")
    print("   - Proses ini akan menghapus embeddings lama")
    print("   - Backup otomatis akan dibuat")
    print("   - Proses mungkin memakan waktu beberapa menit")

    konfirmasi = input("\nLanjutkan regenerate embeddings? (y/n): ").lower()

    if konfirmasi != 'y':
        print("❌ Regenerate dibatalkan")
        return

    print("\n🔄 Memulai proses regenerate...")

    backup_path = backup_existing_embeddings()

    try:
        print("\n🧠 Generating new embeddings...")
        generate_main()

        print("\n🎉 REGENERATE BERHASIL!")
        print("✅ Embeddings baru telah dibuat")

        if backup_path:
            print(f"💾 Backup lama tersimpan di: {backup_path}")

        print("\n💡 Sistem siap digunakan dengan embeddings terbaru")

    except Exception as e:
        print(f"\n❌ Error during regenerate: {e}")

        if backup_path and os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, "models/embeddings.pkl")
                print("🔙 Embeddings lama telah di-restore")
            except:
                print("❌ Gagal restore backup")

if __name__ == "__main__":
    main()