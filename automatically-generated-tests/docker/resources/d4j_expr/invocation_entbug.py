# coding: utf-8
import subprocess
from multiprocessing import Pool, cpu_count
import os
import shlex

def generate_and_localize(seed, project, start, end):

    # excluded = {
    #     'Lang': [2, 23, 56],
    #     'Chart': [4],  # skip Chart-4
    #     'Time': [21],
    #     'Math': [],
    #     'Closure': [63, 93]      
    # }
    excluded = {
        'Lang': [2, 12,23, 56, 65,10,30,41],
        'Chart': [4],
        'Time': [5, 21, 22],  # FIXME: remove 27
        'Math': [15,16,17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,80,81,98,100,101,102,63,54,59],
        'Closure': [63, 93, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 28, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 105, 107, 112, 116, 117, 118, 123, 125, 127, 129] 
    }

    ts_id = f"entbug_TS_{seed}"
    fail_dict = {"generate":[], "localize":[]}

    for version in range(start, end + 1):
        if version in excluded[project]:
            continue
        cp1 = subprocess.run(         # generate tests
            shlex.split("bash generate_test_entbug.sh {} {} evosuite {} 120 {}".format(   ### Note: remember to change the suite name 
            project,
            version,
            ts_id,
            seed
            )),
            cwd="/root/workspace",
            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        print(cp1.stdout, cp1.stderr)
        if cp1.returncode == 0:
            print("Successfully generate tests!")
        else:
            print("Failed to generate tests for {}-{}!".format(project, version))
            fail_dict["generate"].append("{}-{}".format(project, version))
            continue
            # exit(1)

        cp2 = subprocess.run(         # do fault localization
            shlex.split("python3.6 tfd_main_parellel.py {} {} --tool evosuite --id {} --budget 10 --selection TfD_network --noise 0.0".format(
            project,
            version,
            ts_id
            )),
            cwd="/root/workspace",
            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        print(cp2.stdout, cp2.stderr)
        if cp2.returncode == 0:
            print("Successfully localize!")
        else:
            print("Failed to localize for {}-{}!".format(project, version))
            fail_dict["localize"].append("{}-{}".format(project, version))
            continue
            # exit(1)
            
    print("Fail dict:")
    print(fail_dict)

if __name__ == "__main__":

    projects = {
        'Lang':    (1, 65),
        'Chart':   (1, 26),
        'Time':    (1, 27),
        'Math':    (1, 106),
        'Closure': (1, 133)
    }
    excluded = {
        'Lang': [2, 12,23, 56, 65,10,30,41],
        'Chart': [4],
        'Time': [5, 21, 22],  # FIXME: remove 27
        'Math': [15,16,17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,80,81,98,100,101,102,63,54,59],
        # 'Closure': [63, 93,2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 105, 107, 112, 116, 117, 118, 125, 129]      
        'Closure': [63, 93, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 28, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 105, 107, 112, 116, 117, 118, 123, 125, 127, 129] 
    }

    seed = 0
    project = "Closure"
    _, version_num = projects[project]
    process_num = 22

    real_list = []
    for i in range(1, version_num+1):
        if i not in excluded[project]:
            real_list.append(i)

    p = Pool(process_num)
    for i in range(process_num):
        start = i * (len(real_list) // process_num)
        if i == process_num - 1:
            end = len(real_list) - 1
        else:
            end = start + (len(real_list) // process_num) - 1
        p.apply_async(generate_and_localize, args=(seed, project, real_list[start], real_list[end]))

    p.close()
    p.join()



