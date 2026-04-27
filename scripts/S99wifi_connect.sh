#!/bin/sh
#
# S99wifi_connect - Boot-time WiFi station auto-connect
# Connects to hotspot "ROS" using wpa_supplicant + udhcpc
#

WLAN="wlan0"
WPA_CONF="/etc/wpa_supplicant.conf"
DELAY_SEC=3

start() {
    echo "[WiFi] Starting station mode on $WLAN..."

    # Kill any existing wpa_supplicant / udhcpc
    killall wpa_supplicant 2>/dev/null
    killall udhcpc 2>/dev/null
    sleep 1

    # Bring interface up
    ifconfig $WLAN up
    sleep $DELAY_SEC

    # Start wpa_supplicant (daemon mode)
    echo "[WiFi] Connecting to hotspot ROS..."
    wpa_supplicant -B -i $WLAN -c $WPA_CONF
    sleep 2

    # Set static IP (avoid DHCP so you always know the board's address)
    STATIC_IP="172.20.10.100"
    STATIC_GW="172.20.10.1"
    echo "[WiFi] Setting static IP: $STATIC_IP ..."
    ifconfig $WLAN $STATIC_IP netmask 255.255.255.0 up
    route add default gw $STATIC_GW

    echo "[WiFi] Done. Board IP = $STATIC_IP"
    ifconfig $WLAN
}

stop() {
    echo "[WiFi] Stopping WiFi..."
    killall udhcpc 2>/dev/null
    killall wpa_supplicant 2>/dev/null
    ifconfig $WLAN down
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
