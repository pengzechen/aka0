#!/bin/sh
#
# S98tennis - Boot-time auto-start tennis ball tracker
#

TENNIS_BIN="/root/tennis"
TENNIS_MODEL="/root/tennis_cv181x_bf16.cvimodel"
TENNIS_LOG="/root/tennis.log"
VI_CHANNEL=0
export LD_LIBRARY_PATH=/usr/bin/lib:$LD_LIBRARY_PATH

start() {
    echo "[tennis] Starting..."

    # 等待串口就绪
    echo "[tennis] Waiting for serial port /dev/ttyS2..."
    for i in $(seq 1 30); do
        if [ -c /dev/ttyS2 ]; then
            echo "[tennis] Serial port ready"
            break
        fi
        sleep 1
    done

    if [ ! -c /dev/ttyS2 ]; then
        echo "[tennis] ERROR: Serial port /dev/ttyS2 not available after 30s"
        return 1
    fi

    if [ ! -f "$TENNIS_BIN" ]; then
        echo "[tennis] Binary not found: $TENNIS_BIN"
        return 1
    fi
    if [ ! -f "$TENNIS_MODEL" ]; then
        echo "[tennis] Model not found: $TENNIS_MODEL"
        return 1
    fi
    $TENNIS_BIN $TENNIS_MODEL $VI_CHANNEL > $TENNIS_LOG 2>&1 &
    echo "[tennis] Started (PID: $!)"
}

stop() {
    echo "[tennis] Stopping..."
    killall tennis 2>/dev/null
    # 等待进程退出，确保 VPSS 资源释放
    sleep 1
    echo "[tennis] Stopped"
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

exit 0
