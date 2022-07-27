import sys
import time
class Logger(object):

    def __init__(self, loggfile):

        self.terminal = sys.stdout
        self.loggfile = open(loggfile, "a")

        log_sta = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.terminal.write("FEM_taichi 记录时间：  {:s}\n".format(log_sta))
        self.loggfile.write("FEM_taichi 记录时间：  {:s}\n".format(log_sta))

    def write(self, message):

        self.terminal.write(message)
        self.loggfile.write(message)

    def flush(self):
        pass
