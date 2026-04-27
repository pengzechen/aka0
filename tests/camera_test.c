/*
 * ESP32-CAM test tool via /dev/cvi-camera ioctl interface (StarryOS)
 *
 * On StarryOS, /dev/cvi-camera is a kernel device that internally handles
 * all SLIP/CRC/UART3 communication. read() and write() are no-ops;
 * everything goes through ioctl commands:
 *   cmd 1 = INIT       (init camera + ping)
 *   cmd 2 = GET_INFO   (writes CameraInfo to arg pointer)
 *   cmd 3 = GET_FRAME  (writes frame bytes to arg pointer, returns length)
 *
 * Usage:
 *   ./camera_test init
 *   ./camera_test info
 *   ./camera_test frame [--out /tmp/test.jpg]
 *   ./camera_test bench [-n 10]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <time.h>
#include <getopt.h>

#define DEFAULT_DEV        "/dev/cvi-camera"
#define FRAME_MAX_SIZE      (2 * 1024 * 1024)

/* ioctl commands matching StarryOS kernel CviCamera (cvi_camera.rs) */
#define CVICAM_INIT      1
#define CVICAM_GET_INFO  2
#define CVICAM_GET_FRAME 3

/* Must match kernel CameraInfo struct layout */
struct CameraInfo {
    uint16_t width;
    uint16_t height;
    uint8_t format;
    uint8_t connected;
};

/* ---- Commands ---- */

static int do_init(int fd)
{
    if (ioctl(fd, CVICAM_INIT, 0) < 0) {
        perror("ioctl(INIT)");
        return 1;
    }
    printf("[init] OK\n");
    return 0;
}

static int do_info(int fd)
{
    struct CameraInfo info;
    memset(&info, 0, sizeof(info));
    if (ioctl(fd, CVICAM_GET_INFO, &info) < 0) {
        perror("ioctl(GET_INFO)");
        return 1;
    }
    printf("[info] %ux%u  format=%u  connected=%u\n",
           info.width, info.height, info.format, info.connected);
    return 0;
}

static int do_frame(int fd, const char *out_path)
{
    uint8_t *buf = malloc(FRAME_MAX_SIZE);
    if (!buf) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int len = ioctl(fd, CVICAM_GET_FRAME, buf);
    if (len < 0) {
        perror("ioctl(GET_FRAME)");
        free(buf);
        return 1;
    }

    gettimeofday(&t1, NULL);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
    printf("[frame] received %d bytes in %ldms\n", len, ms);

    /* Ensure output directory exists */
    {
        char *dir = strdup(out_path);
        char *slash = strrchr(dir, '/');
        if (slash) {
            *slash = 0;
            mkdir(dir, 0755);
        }
        free(dir);
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        perror("fopen");
        free(buf);
        return 1;
    }
    fwrite(buf, 1, len, f);
    fclose(f);
    free(buf);
    printf("[frame] saved to %s\n", out_path);
    return 0;
}

static int do_bench(int fd, int count)
{
    printf("[bench] capturing %d frames...\n", count);
    int ok = 0, fail = 0;
    struct timeval t_start;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < count; i++) {
        uint8_t *buf = malloc(FRAME_MAX_SIZE);
        if (!buf) { fail++; continue; }

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        int len = ioctl(fd, CVICAM_GET_FRAME, buf);
        if (len < 0) {
            fail++;
            printf("[%d/%d] FAILED\n", i + 1, count);
            free(buf);
            continue;
        }
        free(buf);

        gettimeofday(&t1, NULL);
        long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
        ok++;

        long total_ms = (t1.tv_sec - t_start.tv_sec) * 1000 +
                        (t1.tv_usec - t_start.tv_usec) / 1000;
        float fps_inst = 1000.0f / ms;
        float fps_avg = (i + 1) * 1000.0f / total_ms;

        printf("[%d/%d] %d bytes  %ldms  FPS: %.1f  avg: %.1f\n",
               i + 1, count, len, ms, fps_inst, fps_avg);
    }

    printf("\n[bench] done: %d ok, %d failed\n", ok, fail);
    return fail > 0 ? 1 : 0;
}

/* ---- CLI ---- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "ESP32-CAM test tool (ioctl mode for StarryOS)\n\n"
        "Usage: %s [options] <command>\n\n"
        "Options:\n"
        "  --dev DEV       Device (default: %s)\n"
        "  --out PATH      Output path for frame (default: /tmp/esp_frame.jpg)\n"
        "  -n COUNT        Frame count for bench (default: 10)\n"
        "\n"
        "Commands:\n"
        "  init            Initialize camera\n"
        "  info            Get camera info\n"
        "  frame           Capture frame as JPEG\n"
        "  bench           Benchmark frame capture\n",
        prog, DEFAULT_DEV);
}

int main(int argc, char **argv)
{
    const char *dev = DEFAULT_DEV;
    const char *out_path = "/tmp/esp_frame.jpg";
    int bench_count = 10;

    static struct option long_opts[] = {
        {"dev",   required_argument, NULL, 'd'},
        {"out",   required_argument, NULL, 'o'},
        {"count", required_argument, NULL, 'n'},
        {"help",  no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:o:n:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'd': dev = optarg; break;
        case 'o': out_path = optarg; break;
        case 'n': bench_count = atoi(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    if (optind >= argc) {
        usage(argv[0]);
        return 1;
    }
    const char *cmd = argv[optind];

    printf("=== ESP32-CAM Test (ioctl) ===\n");
    printf("Device: %s\n", dev);

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    int ret;
    if (strcmp(cmd, "init") == 0)
        ret = do_init(fd);
    else if (strcmp(cmd, "info") == 0)
        ret = do_info(fd);
    else if (strcmp(cmd, "frame") == 0)
        ret = do_frame(fd, out_path);
    else if (strcmp(cmd, "bench") == 0)
        ret = do_bench(fd, bench_count);
    else {
        fprintf(stderr, "unknown command: %s\n", cmd);
        ret = 1;
    }

    close(fd);
    return ret;
}
