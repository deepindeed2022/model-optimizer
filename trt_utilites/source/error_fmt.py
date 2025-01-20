#-*- encoding: utf-8 -*-
# Copyright 2025 deepindeed team. All rights reserved.

def fmt_color(s, color="r"):
    pat_dict = {
        "r": "\033[31m {} \033[0m", # 红色
        "g": "\033[32m {} \033[0m",  # 绿色
        "y": "\033[33m {} \033[0m", # 黄色字
        "b": "\033[34m {} \033[0m", # 蓝色字
        # s = "\033[30m 黑色字 \033[0m".replace("黑色字", s)
        # "\033[35m 紫色字 \033[0m"
        # "\033[36m 天蓝字 \033[0m"
        # "\033[37m 白色字 \033[0m"
    }
    return pat_dict[color].format(s)
    
for c in "rgby":
    print(fmt_color("颜色", color=c))