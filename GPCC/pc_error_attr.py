import os, sys
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from data_utils.geometry.inout import read_ply_o3d, write_ply_o3d
import subprocess
import time
import numpy as np
import open3d as o3d
rootdir_tmc13 = os.path.split(__file__)[0]

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def pc_error(infile1, infile2, resolution, normal=False, attr=False, show=False, details=False):
    """
    增加了 attr 参数，用于控制是否计算属性(反射率)的 PSNR
    """
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    if details: 
        headers += ["mse1      (p2point)", "mse1,PSNR (p2point)",
                    "mse2      (p2point)", "mse2,PSNR (p2point)"]

    command = str(rootdir_tmc13+'/pc_error_d' + 
                  ' -a '+infile1+ 
                  ' -b '+infile2+ 
                  ' --hausdorff=1 '+ 
                  ' --resolution='+str(resolution))
                  
    if normal:
        headers +=["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)
        
    # ================= 属性修改部分 =================
    if attr:
        command = str(command + ' -c 1') # 换回 -c 1
        headers += ["c[0],PSNRF", "c[0],    mseF"]
    # ================================================

    results = {}   
    subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    
    while c:
        line = c.decode(encoding='utf-8')
        if show: print(line.strip())
        
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline() 

    # 将 c[0] 重命名为 Reflectance
    if "c[0],PSNRF" in results:
        results["Reflectance_PSNR"] = results.pop("c[0],PSNRF")
    if "c[0],    mseF" in results:
        results["Reflectance_MSE"] = results.pop("c[0],    mseF")

    return results