import mysql.connector
from datetime import datetime
import os

class Database:
    def __init__(self):
        """Initialize MySQL database connection"""
        try:
            self.connection = mysql.connector.connect(
                host='if.unismuh.ac.id',
                user='root',
                password='mariabelajar',
                port=3388,
                database='absensi_mahasiswa_kiki',
                autocommit=True,
                connection_timeout=30,
                auth_plugin='mysql_native_password',
                raise_on_warnings=False,
                use_unicode=True,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            self.cursor = self.connection.cursor()
            print("✅ MySQL Database connected successfully!")

            self._create_tables()

        except mysql.connector.Error as e:
            print(f"❌ MySQL connection error: {e}")
            self.connection = None
            self.cursor = None

    def _create_tables(self):
        """DON'T create new tables - use existing database structure"""
        try:
            self.cursor.execute("SHOW TABLES")
            existing_tables = self.cursor.fetchall()
            print("🔍 Existing tables in database:")
            for table in existing_tables:
                print(f"   - {table[0]}")

            print("✅ Using existing database structure!")

        except mysql.connector.Error as e:
            print(f"❌ Database check error: {e}")

    def ensure_connection(self):
        """Ensure database connection is alive"""
        try:
            if not self.connection or not self.connection.is_connected():
                print("🔄 Database connection lost, reconnecting...")
                self.__init__()
                return self.connection is not None
            return True
        except:
            print("🔄 Connection check failed, reconnecting...")
            self.__init__()
            return self.connection is not None

    def _insert_sample_data(self):
        """DON'T insert sample data - use existing database"""
        print("🔧 Using existing database data - no sample data needed!")
        pass

    def create_session(self, dosen_name, mata_kuliah):
        """Create new attendance session using REAL database structure"""
        try:
            # Reconnect jika connection hilang
            if not self.connection or not self.connection.is_connected():
                print("🔄 Reconnecting to database...")
                self.__init__()
                
            if not self.connection:
                print("❌ Cannot establish database connection")
                return None

            from datetime import date, time
            today = date.today()
            current_time = time(8, 0)
            end_time = time(10, 0)

            query = """INSERT INTO sessions 
                      (dosen, mata_kuliah, tanggal, waktu_mulai, waktu_selesai, status) 
                      VALUES (%s, %s, %s, %s, %s, %s)"""

            print(f"🔥 Creating session: {dosen_name}, {mata_kuliah}, {today}")
            self.cursor.execute(query, (dosen_name, mata_kuliah, today, current_time, end_time, 'active'))
            self.connection.commit()

            session_id = self.cursor.lastrowid
            print(f"✅ Session created with ID: {session_id}")
            return session_id

        except mysql.connector.Error as e:
            print(f"❌ Session creation error: {e}")
            # Coba reconnect dan retry sekali
            try:
                print("🔄 Trying to reconnect and retry...")
                self.__init__()
                if self.connection:
                    self.cursor.execute(query, (dosen_name, mata_kuliah, today, current_time, end_time, 'active'))
                    self.connection.commit()
                    session_id = self.cursor.lastrowid
                    print(f"✅ Session created on retry with ID: {session_id}")
                    return session_id
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
            return None

    def add_attendance(self, session_id, student_name, confidence):
        """Add attendance record using REAL database structure with proper NIM lookup"""
        try:
            if not self.ensure_connection():
                print("❌ No database connection available")
                return False

            print(f"🔍 Checking if {student_name} already recorded in session {session_id}")
            check_query = "SELECT COUNT(*) FROM attendance WHERE session_id = %s AND nama_mahasiswa = %s"
            self.cursor.execute(check_query, (session_id, student_name))
            count = self.cursor.fetchone()[0]

            if count > 0:
                print(f"⚠️ Student {student_name} already recorded in this session")
                return False

            # Get NIM from mahasiswa table
            nim = self.get_student_nim(student_name)
            print(f"📝 Found NIM for {student_name}: {nim}")

            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M:%S")  # Current time

            query = """INSERT INTO attendance 
                      (session_id, nama_mahasiswa, nim, waktu_hadir, confidence, status) 
                      VALUES (%s, %s, %s, %s, %s, %s)"""

            print(f"💾 Inserting attendance: session_id={session_id}, name={student_name}, nim={nim}, confidence={confidence}")
            self.cursor.execute(query, (session_id, student_name, nim, current_time, confidence, 'hadir'))
            self.connection.commit()

            # Sleep sebentar untuk memastikan commit berhasil
            import time
            time.sleep(0.1)

            # Verify insertion
            verify_query = "SELECT COUNT(*) FROM attendance WHERE session_id = %s AND nama_mahasiswa = %s"
            self.cursor.execute(verify_query, (session_id, student_name))
            verify_count = self.cursor.fetchone()[0]
            
            print(f"✅ Attendance recorded successfully! Record count: {verify_count}")
            print(f"📊 SUCCESS: {student_name} attendance saved to database")
            return True

        except mysql.connector.Error as e:
            print(f"❌ Add attendance error: {e}")
            return False

            print(f"✅ Attendance recorded: {student_name} (NIM: {nim}, confidence: {confidence:.3f})")
            return True

        except mysql.connector.Error as e:
            print(f"❌ Attendance recording error: {e}")
            return False

    def get_session_attendance(self, session_id):
        """Get attendance for specific session using REAL database structure with proper NIM - ENHANCED DEBUG"""
        try:
            if not self.connection:
                print("❌ No database connection for get_session_attendance")
                return []

            print(f"🔍 Getting attendance for session {session_id}")
            print(f"🔍 Connection status: {self.connection is not None}")

            # Debug: cek total records di tabel attendance
            debug_query = "SELECT COUNT(*) FROM attendance WHERE session_id = %s"
            self.cursor.execute(debug_query, (session_id,))
            total_count = self.cursor.fetchone()[0]
            print(f"🔍 Total attendance records for session {session_id}: {total_count}")

            # Query dengan JOIN untuk mendapatkan NIM dari tabel mahasiswa
            query = """
            SELECT 
                a.nama_mahasiswa, 
                COALESCE(m.nim, a.nim, 'N/A') as nim,
                a.confidence, 
                a.waktu_hadir, 
                a.created_at, 
                a.status 
            FROM attendance a
            LEFT JOIN mahasiswa m ON a.nama_mahasiswa = m.nama
            WHERE a.session_id = %s 
            ORDER BY a.created_at ASC
            """
            
            print(f"🔍 Executing query with session_id: {session_id}")
            self.cursor.execute(query, (session_id,))
            results = self.cursor.fetchall()

            print(f"🔍 Raw SQL results: {len(results)} rows")
            for i, row in enumerate(results):
                print(f"   Row {i+1}: {row}")

            attendance_list = []
            for row in results:
                attendance_record = {
                    'student_name': row[0],    # nama_mahasiswa
                    'nama_mahasiswa': row[0],  # alias untuk kompatibilitas
                    'nim': row[1] if row[1] and row[1] != 'N/A' else 'N/A',  # nim dari mahasiswa atau attendance
                    'confidence': float(row[2]) if row[2] else 0.0,      # confidence
                    'waktu_hadir': row[3],     # waktu_hadir
                    'timestamp': row[4],       # created_at
                    'created_at': row[4],      # alias untuk kompatibilitas
                    'status': row[5] if row[5] else 'hadir'       # status
                }
                attendance_list.append(attendance_record)
                print(f"   Processed: {attendance_record}")

            print(f"📊 Database: Found {len(attendance_list)} attendance records for session {session_id}")
            return attendance_list

        except mysql.connector.Error as e:
            print(f"❌ Get attendance error: {e}")
            return []

    def get_attendance_by_session(self, session_id):
        """Alias untuk get_session_attendance untuk kompatibilitas"""
        return self.get_session_attendance(session_id)

    def get_student_nim(self, student_name):
        """Get student NIM by name from mahasiswa table"""
        try:
            if not self.connection:
                return 'N/A'

            query = "SELECT nim FROM mahasiswa WHERE nama = %s LIMIT 1"
            self.cursor.execute(query, (student_name,))
            result = self.cursor.fetchone()
            
            if result:
                return result[0] if result[0] else 'N/A'
            else:
                print(f"⚠️ NIM not found for student: {student_name}")
                return 'N/A'

        except mysql.connector.Error as e:
            print(f"❌ Error getting student NIM: {e}")
            return 'N/A'

    def get_dosen_all(self):
        """Get all dosens for dropdown using REAL database structure"""
        try:
            if not self.connection:
                return []

            print("🔥 DEBUG: Fetching dosens...")

            possible_queries = [
                "SELECT nama FROM dosen ORDER BY nama",
                "SELECT nama_dosen FROM dosen ORDER BY nama_dosen", 
                "SELECT name FROM dosen ORDER BY name",
                "SELECT * FROM dosen"
            ]

            for query in possible_queries:
                try:
                    self.cursor.execute(query)
                    results = self.cursor.fetchall()
                    print(f"🔥 Query '{query}' result: {results}")

                    if results:
                        if query.startswith("SELECT *"):
                            columns = [desc[0] for desc in self.cursor.description]
                            print(f"🔥 Dosen table columns: {columns}")

                            for col_idx, col_name in enumerate(columns):
                                if 'nama' in col_name.lower() or 'name' in col_name.lower():
                                    print(f"🔥 Using column: {col_name}")
                                    return [row[col_idx] for row in results if row[col_idx]]

                            for row in results:
                                for col_idx, value in enumerate(row):
                                    if isinstance(value, str) and value.strip():
                                        print(f"🔥 Using first string column (index {col_idx})")
                                        return [r[col_idx] for r in results if r[col_idx]]
                        else:
                            return [row[0] for row in results if row[0]]

                except mysql.connector.Error as e:
                    print(f"⚠️ Query '{query}' failed: {e}")
                    continue

            print("❌ No valid dosen data found")
            return []

        except mysql.connector.Error as e:
            print(f"❌ Get dosen error: {e}")
            return []

    def get_mata_kuliah_all(self):
        """Get all mata kuliah for dropdown using REAL database structure"""
        try:
            if not self.connection:
                return []

            print("🔥 DEBUG: Fetching mata kuliah...")

            possible_queries = [
                "SELECT nama FROM mata_kuliah ORDER BY nama",
                "SELECT name FROM mata_kuliah ORDER BY name", 
                "SELECT DISTINCT matkul FROM dosen ORDER BY matkul",
                "SELECT * FROM mata_kuliah LIMIT 5"
            ]

            for query in possible_queries:
                try:
                    self.cursor.execute(query)
                    results = self.cursor.fetchall()
                    print(f"🔥 Query '{query}' result: {results}")

                    if results:
                        if query.startswith("SELECT *"):
                            columns = [desc[0] for desc in self.cursor.description]
                            print(f"🔥 Mata kuliah table columns: {columns}")

                            for col_idx, col_name in enumerate(columns):
                                if 'nama' in col_name.lower() or 'name' in col_name.lower():
                                    print(f"🔥 Using column: {col_name}")
                                    return [row[col_idx] for row in results if row[col_idx]]

                            for row in results:
                                for col_idx, value in enumerate(row):
                                    if isinstance(value, str) and value.strip():
                                        print(f"🔥 Using first string column (index {col_idx})")
                                        return [r[col_idx] for r in results if r[col_idx]]
                        else:
                            return [row[0] for row in results if row[0]]

                except mysql.connector.Error as e:
                    print(f"⚠️ Query '{query}' failed: {e}")
                    continue

            print("❌ No valid mata kuliah data found")
            return []

        except mysql.connector.Error as e:
            print(f"❌ Get mata kuliah error: {e}")
            return []

    def close(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("✅ Database connection closed")
        except:
            pass

if __name__ == "__main__":
    print("🔧 Testing Database connection...")
    db = Database()

    if db.connection:
        print("✅ Database test successful!")

        dosens = db.get_dosen_all()
        subjects = db.get_mata_kuliah_all()

        print(f"📋 Available dosens: {len(dosens)}")
        for i, dosen in enumerate(dosens, 1):
            print(f"   {i}. {dosen}")

        print(f"📚 Available subjects: {len(subjects)}")
        for i, subject in enumerate(subjects, 1):
            print(f"   {i}. {subject}")

        try:
            print("🔍 Checking existing table structures...")

            db.cursor.execute("DESCRIBE sessions")
            sessions_columns = db.cursor.fetchall()
            print("� Sessions table structure:")
            for col in sessions_columns:
                print(f"   - {col[0]} ({col[1]})")

            db.cursor.execute("DESCRIBE attendance")
            attendance_columns = db.cursor.fetchall()
            print("� Attendance table structure:")
            for col in attendance_columns:
                print(f"   - {col[0]} ({col[1]})")

        except Exception as e:
            print(f"⚠️ Cannot check table structure: {e}")

        try:
            db.cursor.execute("SELECT * FROM sessions LIMIT 1")
            if db.cursor.description:
                columns = [desc[0] for desc in db.cursor.description]
                print(f"� Sessions columns: {columns}")

        except Exception as e:
            print(f"⚠️ Cannot check sessions structure: {e}")

        try:
            db.cursor.execute("SELECT * FROM attendance LIMIT 1")
            if db.cursor.description:
                columns = [desc[0] for desc in db.cursor.description]
                print(f"🔧 Attendance columns: {columns}")

        except Exception as e:
            print(f"⚠️ Cannot check attendance structure: {e}")

        db.close()
    else:
        print("❌ Database test failed!")
