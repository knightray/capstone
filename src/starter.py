import os
import define

define.init_log()
define.log("========Start trainning=========")
os.system("python training.py")
define.log("========Start verify   =========")
os.system("python test.py --type=epoch")
define.close_log()
