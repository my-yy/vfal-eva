import subprocess
import os
from concurrent.futures import wait, ProcessPoolExecutor
import time
import glob


def core(index):
    input_file = all_mp4[index]
    assert os.path.exists(input_file)
    output_file = os.path.join(save_root, os.path.basename(input_file)) + ".wav"

    cmd = "ffmpeg -y -i %s -ab 160k -ac 1 -ar 16000 -vn %s" % (input_file, output_file)
    ans = subprocess.call(cmd, shell=True)
    if ans != 0:
        print("error:", input_file, ans)

    if index > 0 and index % 100 == 0:
        time_cost = time.time() - start_time
        percentage = index / len(all_mp4)
        total_time = time_cost / percentage / 3600
        print(index, "total time:", total_time, "progress:", percentage)
    return output_file


if __name__ == "__main__":
    all_mp4 = glob.glob("data/test/*.mp4")
    save_root = "wav_output/"
    pool_size = 8

    print("start！")
    start_time = time.time()
    pool = ProcessPoolExecutor(pool_size)
    tasks = [pool.submit(core, i) for i in range(len(all_mp4))]
    wait(tasks)
    print('done')
