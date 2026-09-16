// Minimal V4L2 mmap-streaming capture, exposed as a small C API for
// dart:ffi (see ../lib/src/v4l2_bindings.dart). Implements the standard
// V4L2 capture sequence documented in the Linux kernel's own V4L2 API
// docs (Documentation/userspace-api/media/v4l/dev-capture.rst, and the
// buffer-queue mechanics in v4l2-mmap.rst / v4l2-buffer.rst): open the
// device -> negotiate a format with VIDIOC_S_FMT -> request mmap-backed
// buffers with VIDIOC_REQBUFS -> mmap + queue each with VIDIOC_QBUF ->
// VIDIOC_STREAMON -> poll()/VIDIOC_DQBUF/VIDIOC_QBUF in a loop ->
// VIDIOC_STREAMOFF -> munmap -> close. These ioctls/structs are the
// public Linux kernel UAPI (<linux/videodev2.h>), not third-party code —
// this file is an original implementation of that documented sequence,
// not copied from any example.
//
// S5b (doc/KR/linux_port_scope.md): replaces the S5a ffmpeg PoC so the
// shipped binary has no GPL dependency.

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

#define FACEKIT_V4L2_NUM_BUFFERS 4

typedef struct {
    void *start;
    size_t length;
} MappedBuffer;

typedef struct {
    int fd;
    uint32_t width;
    uint32_t height;
    uint32_t pixelformat;
    uint32_t frame_size; // sizeimage as negotiated by the driver
    MappedBuffer buffers[FACEKIT_V4L2_NUM_BUFFERS];
    int buffer_count;
    int streaming;
} V4l2Capture;

static int xioctl(int fd, unsigned long request, void *arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

void v4l2_close(V4l2Capture *capture);

// Opens [device], negotiates [width]x[height] at [pixelformat] (a
// V4L2_PIX_FMT_* fourcc, e.g. V4L2_PIX_FMT_YUYV), and starts streaming.
// Returns NULL on any failure (bad device, format rejected, mmap failed,
// ...) — check errno-style failures by retrying with different
// width/height/format if this returns NULL. The driver may adjust
// width/height to the nearest supported size; read them back with
// v4l2_get_width/v4l2_get_height rather than assuming the requested
// values stuck.
V4l2Capture *v4l2_open(const char *device, uint32_t width, uint32_t height, uint32_t pixelformat) {
    int fd = open(device, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) return NULL;

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0 ||
        !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(cap.capabilities & V4L2_CAP_STREAMING)) {
        close(fd);
        return NULL;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        close(fd);
        return NULL;
    }
    if (fmt.fmt.pix.pixelformat != pixelformat) {
        // Driver silently substituted a different format than requested —
        // treat as failure rather than handing the caller frames in a
        // format it didn't ask for.
        close(fd);
        return NULL;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = FACEKIT_V4L2_NUM_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 1) {
        close(fd);
        return NULL;
    }

    V4l2Capture *capture = calloc(1, sizeof(V4l2Capture));
    if (!capture) {
        close(fd);
        return NULL;
    }
    capture->fd = fd;
    capture->width = fmt.fmt.pix.width;
    capture->height = fmt.fmt.pix.height;
    capture->pixelformat = fmt.fmt.pix.pixelformat;
    capture->frame_size = fmt.fmt.pix.sizeimage;
    capture->buffer_count = (int)req.count;

    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            v4l2_close(capture);
            return NULL;
        }
        capture->buffers[i].length = buf.length;
        capture->buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (capture->buffers[i].start == MAP_FAILED) {
            capture->buffers[i].start = NULL;
            v4l2_close(capture);
            return NULL;
        }
        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            v4l2_close(capture);
            return NULL;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        v4l2_close(capture);
        return NULL;
    }
    capture->streaming = 1;

    return capture;
}

uint32_t v4l2_get_width(V4l2Capture *capture) { return capture->width; }
uint32_t v4l2_get_height(V4l2Capture *capture) { return capture->height; }
uint32_t v4l2_get_frame_size(V4l2Capture *capture) { return capture->frame_size; }

// Blocks (via poll()) up to [timeout_ms] for a frame, then copies it into
// [out_buffer] (caller-owned, must be >= v4l2_get_frame_size() bytes —
// copying, rather than handing back a pointer into the mmap'd buffer,
// keeps buffer lifetime entirely on the Dart side simple: the mmap
// region is immediately re-queued and can be overwritten by the driver
// at any time after this call returns).
//
// [out_sequence], if non-NULL, receives the driver's per-frame sequence
// counter (struct v4l2_buffer.sequence) — increments by exactly 1 per
// captured frame, so the caller can detect drops (a gap) by comparing
// consecutive values, independent of this function's own timing.
//
// Returns the number of bytes copied (> 0) on success, 0 on timeout,
// negative on error.
int32_t v4l2_read_frame(V4l2Capture *capture, uint8_t *out_buffer, int32_t timeout_ms, uint32_t *out_sequence) {
    struct pollfd pfd;
    pfd.fd = capture->fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int pr = poll(&pfd, 1, timeout_ms);
    if (pr < 0) return -1;
    if (pr == 0) return 0; // timeout, no frame ready

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(capture->fd, VIDIOC_DQBUF, &buf) < 0) {
        return -2;
    }
    if ((int)buf.index >= capture->buffer_count) {
        // Shouldn't happen (driver returned an index we never registered),
        // but re-queue defensively and report failure rather than reading
        // out of bounds.
        xioctl(capture->fd, VIDIOC_QBUF, &buf);
        return -4;
    }

    uint32_t n = buf.bytesused;
    memcpy(out_buffer, capture->buffers[buf.index].start, n);
    if (out_sequence) *out_sequence = buf.sequence;

    if (xioctl(capture->fd, VIDIOC_QBUF, &buf) < 0) {
        return -3;
    }

    return (int32_t)n;
}

void v4l2_close(V4l2Capture *capture) {
    if (!capture) return;
    if (capture->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(capture->fd, VIDIOC_STREAMOFF, &type);
    }
    for (int i = 0; i < capture->buffer_count; i++) {
        if (capture->buffers[i].start) {
            munmap(capture->buffers[i].start, capture->buffers[i].length);
        }
    }
    if (capture->fd >= 0) close(capture->fd);
    free(capture);
}
