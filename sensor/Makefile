# Makefile for sensor_test
# Standalone build without middleware dependency

# 交叉编译工具链
CROSS_COMPILE ?= /home/ajax/Projects/sg2002/LicheeRV-Nano-Build/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-

CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar

# 目录定义
SENSOR_DIR = $(PWD)
INCLUDE_DIR = $(SENSOR_DIR)/include
LIB_DIR = $(SENSOR_DIR)/lib

# 头文件路径
INCS = -I$(INCLUDE_DIR)
INCS += -I$(INCLUDE_DIR)/isp/sg200x
INCS += -I$(INCLUDE_DIR)/common

# 编译选项
CFLAGS = -Wall -Werror -fPIC
CFLAGS += -Os
CFLAGS += -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d
CFLAGS += $(INCS)

# 链接选项
LDFLAGS = -L$(LIB_DIR)
LDFLAGS += -Wl,--start-group
LDFLAGS += -lsample -lisp -lisp_algo -lae -lawb -laf
LDFLAGS += -lvpu -lcvi_ispd2 -lcvi_bin_isp -lcvi_bin
LDFLAGS += -lsns_full -lsns_gc4653 -lsns_sc035gs -lsns_os04a10 -lsns_lt6911
LDFLAGS += -ljson-c -lsys -lini
LDFLAGS += -Wl,--end-group
LDFLAGS += -lpthread -lm -ldl -latomic

# 源文件和目标文件
SRCS = sensor_test.c ae_test.c
OBJS = $(SRCS:.c=.o)
TARGET = sensor_test

# 默认目标
.PHONY: all clean

all: $(TARGET)

%.o: %.c
	@echo "[CC] $<"
	@$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	@echo "[LD] $@"
	@$(CXX) -o $@ $(OBJS) $(LDFLAGS)
	@echo "✓ 编译完成: $(TARGET)"

clean:
	@rm -f $(OBJS) $(TARGET)
	@echo "清理完成"
