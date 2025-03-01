import argparse
import signal
import subprocess
import time
from statistics import mean
import time

LOG_FILE = None


"""Ctrl + C"""
def exit_pro(signum, frame):
    if LOG_FILE:
        LOG_FILE.close()
    exit()


signal.signal(signal.SIGINT, exit_pro)
signal.signal(signal.SIGTERM, exit_pro)
# where the tegrastats binary file is
BIN_PATH = 'tegrastats'
LOG_FILE_PATH = 'LOGS.csv'


"""
TODO: 1. extend to different format
      2. recognize format automatically
"""
def tegrastats(log_file_path=None, freq=10, board="nano"):
    """@:arg - log file name
       @:arg - sampling frequency"""
    cmds = ["tegrastats", "--interval",  str(int(1000/freq))]
    if log_file_path is None:
      log_file_path = "{}_parsed_tegrastats.csv".format(int(time.time()))
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    try:
      log_file = open(log_file_path, 'a+')
      while True:
        current_stat = p.stdout.readline().decode().strip()
        if current_stat == '':
          raise ValueError("Tegrastats error detected")
        fields = current_stat.split(" ")
        stats, others = {}, {}
        stats["ram_used"], stats["ram_tot"] = [int(e) for e in fields[1][:-2].split("/")]
        others["cpu_perc_freq"] = [[int(e.split("%@")[0]), int(e.split("%@")[1])] for e in fields[5][1:-1].split(",")]
        stats["avg_used"] = mean([perc/100. for perc, freq in others["cpu_perc_freq"]])
        stats["avg_freq"] = mean([freq for perc, freq in others["cpu_perc_freq"]])
        # Weighted average frequency
        stats["w_avg_freq"] = mean([perc/100. *freq for perc, freq in others["cpu_perc_freq"]])
        # External Memory Control Frequency percentage used
        stats["emc"] = int(fields[7][:-1]) 
        stats["gpu_used"] = int(fields[9][:-1])
        for i in [16, 18, 20]:
          stats["pom_5v_{}".format(fields[i].split("_")[-1])] = int(fields[i+1].split("/")[0])
        text = str(time.time()) + "," + ",".join([str(stats[e]) for e in sorted(stats)])
        if write_to_log:
          log_file.write(text + '\n')
    except Exception as e:
      raise e
    finally:
      log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tegrastats log')
    parser.add_argument('-b', '--bin', metavar='where tegrastats is', required=False, dest='bin_path', action='store')
    # the script will not write to log file unless you define the output log file path
    parser.add_argument('-o', '--output', metavar='write the log file to here', required=False, dest='log_file_path',
                        action='store')
    parser.add_argument('-p', '--params', metavar='additional arguments of tegrastats', required=False, dest='your_args',
                        action='store')
    args = parser.parse_args()
    if args.log_file_path:
        LOG_FILE_PATH = args.log_file_path
        write_to_log = True
    else:
        write_to_log = False
    if args.bin_path:
        BIN_PATH = args.bin_path
    work(write_to_log, args.your_args)

