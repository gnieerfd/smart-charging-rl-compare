import pandas as pd
import numpy as np
import os

def process_charging_profile(file_path, output_path, battery_capacity=300.0, time_interval_minutes=15):
    """
    Membaca profil charging dari Excel dan mengubahnya menjadi jadwal bus
    (Arrival Time, Initial SOC) untuk simulasi.
    
    Args:
        file_path: Path ke file Excel
        output_path: Path untuk menyimpan CSV hasil
        battery_capacity: Kapasitas baterai bus dalam kWh (default 300)
        time_interval_minutes: Interval waktu antar baris data dalam menit (default 15)
    """
    print(f"Membaca file: {file_path}")
    
    # Baca Excel, baris ke-2 (index 1) adalah header yang benar berdasarkan inspeksi sebelumnya
    # Row 0 di pandas (baris 1 excel) adalah header CS 1, CS 2...
    try:
        df = pd.read_excel(file_path, header=None)
        print("Raw DataFrame Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading excel: {e}")
        return

    # Ambil nama kolom dari baris dengan index 1 (CS 1, CS 2, ...)
    # Data dimulai dari index 2
    cs_names = df.iloc[1].values
    data = df.iloc[2:].copy()
    data.columns = cs_names
    
    # Konversi data ke numeric, error jadi NaN lalu diisi 0
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    bus_schedule = []
    
    print(f"Memproses {len(data.columns)} Charging Stations...")
    print(f"Columns found: {data.columns.tolist()[:5]} ...")
    
    for col in data.columns:
        # Ensure column name is string and strip whitespace
        col_str = str(col).strip()
        # print(f"Checking column: '{col_str}'") # Debug
        
        if not col_str.startswith('CS'):
            # print(f"Skipping {col_str}")
            continue
            
        profile = data[col].values
        
        # Cari kapan charging dimulai (nilai > 0 pertama)
        # Kita asumsikan baris merepresentasikan urutan waktu dari awal hari (atau periode data)
        active_indices = np.where(profile > 0.1)[0] # Use threshold > 0.1 to avoid float noise
        
        if len(active_indices) == 0:
            print(f"Warning: {col_str} tidak memiliki aktivitas charging (max val: {np.max(profile)})")
            continue
            
        # Arrival time adalah index pertama charging * interval
        start_idx = active_indices[0]
        arrival_minute = start_idx * time_interval_minutes
        
        # Hitung total energi yang disalurkan (kWh)
        # Coba asumsi interval 1 menit dulu
        energy_1min = np.sum(profile) * (1.0 / 60.0)
        
        # Coba asumsi interval 15 menit
        energy_15min = np.sum(profile) * (15.0 / 60.0)
        
        # Heuristik: Kapasitas bus ~300 kWh. 
        # Jika energy_1min masuk akal (misal < 600 kWh), gunakan 1 menit.
        # Jika energy_15min lebih masuk akal, gunakan 15 menit.
        
        if energy_1min <= battery_capacity * 1.5:
            used_interval = 1
            total_energy_kwh = energy_1min
        else:
            used_interval = 15
            total_energy_kwh = energy_15min
            
        # Override jika user memaksa interval (opsional, tapi di sini kita auto-detect per kolom atau global)
        # Kita gunakan global time_interval_minutes jika diberikan secara eksplisit dan masuk akal?
        # Untuk sekarang, biar script yang tentukan agar aman.
        
        # Arrival time
        arrival_minute = start_idx * used_interval
        
        # Estimasi Initial SOC
        soc_init = 1.0 - (total_energy_kwh / battery_capacity)
        
        # Clip SOC agar masuk akal
        soc_init = np.clip(soc_init, 0.05, 0.95)
        
        bus_schedule.append({
            'bus_id': col_str,
            'arrival_minute': int(arrival_minute),
            'soc_init': round(soc_init, 4),
            'energy_required_kwh': round(total_energy_kwh, 2),
            'capacity': battery_capacity,
            'detected_interval': used_interval
        })
        
    # Buat DataFrame hasil
    if not bus_schedule:
        print("Error: Tidak ada data bus yang berhasil diproses.")
        return

    schedule_df = pd.DataFrame(bus_schedule)
    
    # Urutkan berdasarkan kedatangan
    if 'arrival_minute' in schedule_df.columns:
        schedule_df = schedule_df.sort_values('arrival_minute')
    else:
        print("Error: Kolom arrival_minute tidak ditemukan.")
        return
    
    print("\nSampel 5 data teratas:")
    print(schedule_df.head())
    
    print(f"\nMenyimpan hasil ke: {output_path}")
    schedule_df.to_csv(output_path, index=False)
    print("Selesai.")

if __name__ == "__main__":
    # Path file
    input_file = 'New_Data_50_Charging_Profile.xlsx'
    output_file = 'bus_schedule.csv'

    # Cek apakah file input ada
    if not os.path.exists(input_file):
        print(f"File tidak ditemukan: {input_file}")
    else:
        # Asumsi interval 15 menit (umum untuk data load profile), bisa disesuaikan
        # Jika data per menit, ubah time_interval_minutes=1
        # Kita perlu cek jumlah baris untuk menebak interval
        # Jika 1 hari = 96 baris -> 15 menit. Jika 1440 baris -> 1 menit.
        
        df_check = pd.read_excel(input_file, header=None)
        num_rows = len(df_check) - 1 # kurangi header
        print(f"Jumlah baris data: {num_rows}")
        
        interval = 15 # Default
        if 1400 <= num_rows <= 1500:
            interval = 1
            print("Terdeteksi data per menit (interval = 1 menit)")
        elif 90 <= num_rows <= 100:
            interval = 15
            print("Terdeteksi data per 15 menit (interval = 15 menit)")
        else:
            print(f"Interval waktu tidak pasti (rows={num_rows}), menggunakan default 15 menit.")
            
        process_charging_profile(input_file, output_file, time_interval_minutes=interval)
