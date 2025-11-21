import psutil
import pandas as pd
import time
import os

# ==========================
# CONFIGURATION
# ==========================
SAMPLE_INTERVAL = 2
NUM_SAMPLES = 300
OUTPUT_FILE = "process_metrics2.csv"
SAVE_EVERY = 50

# ==========================
# INITIAL CPU INIT
# ==========================
for p in psutil.process_iter():
    try:
        p.cpu_percent(None)
    except:
        pass

data = []
print(f"Collecting for ~{(SAMPLE_INTERVAL * NUM_SAMPLES)/60:.1f} minutes...")

prev_io = {}  # track read/write speeds

for i in range(NUM_SAMPLES):

    system_cpu = psutil.cpu_percent()
    system_memory = psutil.virtual_memory().percent
    timestamp_now = time.time()

    for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent',
                                  'num_threads', 'io_counters', 'status', 'create_time']):
        try:
            info = p.info
            info['timestamp'] = timestamp_now
            info['system_cpu'] = system_cpu
            info['system_memory'] = system_memory

            # CPU times
            ct = p.cpu_times()
            info['cpu_user_time'] = ct.user
            info['cpu_system_time'] = ct.system

            # Memory details
            mem = p.memory_info()
            info['mem_rss'] = mem.rss
            info['mem_vms'] = mem.vms

            # Process priority + parent
            info["nice"] = p.nice()
            info["ppid"] = p.ppid()

            # IO speed calculation
            io = info.get('io_counters')
            if io:
                last = prev_io.get(info['pid'], (io.read_bytes, io.write_bytes))
                info['read_speed'] = io.read_bytes - last[0]
                info['write_speed'] = io.write_bytes - last[1]
                prev_io[info['pid']] = (io.read_bytes, io.write_bytes)
            else:
                info['read_speed'] = 0
                info['write_speed'] = 0

            # Add unique record id
            info["record_id"] = f"{i}_{info['pid']}"

            info.pop("io_counters", None)
            data.append(info)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Periodic Save
    if (i + 1) % SAVE_EVERY == 0:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, mode='a', index=False,
                  header=not os.path.exists(OUTPUT_FILE))
        data.clear()
        print(f"Saved {i+1}/{NUM_SAMPLES} samples...")

    time.sleep(SAMPLE_INTERVAL)

# Final Save
if data:
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, mode='a', index=False,
              header=not os.path.exists(OUTPUT_FILE))

print("\n Data collection complete!")
