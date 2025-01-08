import builtins
import datetime

# 保存原始的 print 函数
original_print = builtins.print

def custom_print(*args, **kwargs):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # 在输出之前打印当前时间
    original_print(f"[{current_time}] ", end="")
    # 调用原始的 print 函数
    original_print(*args, **kwargs)

# 替换内置的 print 函数
builtins.print = custom_print
