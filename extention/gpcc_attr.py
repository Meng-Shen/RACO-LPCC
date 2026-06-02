# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import subprocess
import time
import os, sys
rootdir_tmc13 = os.path.split(__file__)[0]
rootdir_cfg = os.path.join(os.path.split(__file__)[0], 'gpcc_cfg')


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def gpcc_encode(filedir, bin_dir, posQuantscale=1, attrQP=28, version=22, cfgdir='kitti.cfg', DBG=False):
    cfgdir = os.path.join(rootdir_cfg, cfgdir)
    assert os.path.exists(cfgdir)

    # 恢复使用最兼容的 --qp 参数
    cmd = rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=0 ' \
        + ' --config='+cfgdir \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --attribute=reflectance ' \
        + ' --qp='+str(attrQP) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir

    # 注意这里的修改：合并 stdout 和 stderr，以便捕获报错
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    
    # 捕获所有输出
    output_lines = []
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        output_lines.append(line.strip())
        if DBG: print(line.strip())
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    # 等待进程结束
    subp.wait()

    # === 核心排错逻辑 ===
    # 如果编码器没有成功生成 bin 文件，把 TMC13 的真实报错打印出来！
    if not os.path.exists(bin_dir) or os.path.getsize(bin_dir) == 0:
        print("\n❌ [致命错误] TMC13 编码器崩溃或未能生成 bin 文件！")
        print("====== TMC13 真实输出日志 ======")
        for log_line in output_lines:
            print(log_line)
        print("================================")
        raise RuntimeError("G-PCC Encoder Failed. Check the log above.")

    return results

def gpcc_decode(bin_dir, dec_dir, version=22, DBG=False):
    cmd = rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=1 ' \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # subp.wait()
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if DBG: print(c)   
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()

    return results
