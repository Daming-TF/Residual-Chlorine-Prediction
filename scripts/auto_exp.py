from datetime import date, datetime
import subprocess
import time
import os
# from apscheduler.schedulers.background import BackgroundScheduler


if __name__ == "__main__":
    window_size = [350, 400, 450]# [5, 10, 20, 50, 100, 200, 300]
    label_length = [1, 5, 10, 20, 25, 50, 70, 100, 150, 200, 250, 300, 350]
    for ws in window_size:
        for ll in label_length:
            pn = int(450/ll)
            i = f"python E:\Project\ResidualChlorinePrediction\scripts\main.py --window_size {ws} --label_length {ll} --pred_num {pn}"
            if subprocess.call(i) == 0:
                continue
            time.sleep(15)
            # os.system(f'e:')
            # os.system(
            #     f"python E:\Project\ResidualChlorinePrediction\scripts\main.py --window_size {ws} --label_length {ll}")
            # print(f"sleep! - {datetime.now()}")
            # time.sleep(650)
