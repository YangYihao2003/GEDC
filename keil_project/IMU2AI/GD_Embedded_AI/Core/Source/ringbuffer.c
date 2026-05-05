#include "ringbuffer.h"
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

/* 辅助宏 */
#define MY_MIN(a, b)  ((a) < (b) ? (a) : (b))

/**
 * @brief 初始化环形缓冲区
 */
void rb_init(RingBuffer *rb) {
    rb->write = 0;
    rb->read = 0;
    rb->rb_mutex = xSemaphoreCreateBinary();   // 如果需要线程安全，保留；若已去锁可删除
    xSemaphoreGive(rb->rb_mutex);
}

/**
 * @brief 写入一个字节
 */
bool rb_put(RingBuffer *rb, uint8_t byte) {
    if (rb_is_full(rb)) {
        return false;
    }
    uint32_t w = rb->write;
    rb->data[w] = byte;
    rb->write = (w + 1) & BUFFER_MASK;
    return true;
}

/**
 * @brief 读取一个字节
 */
bool rb_get(RingBuffer *rb, uint8_t *byte) {
    if (rb_is_empty(rb)) {
        return false;
    }
    uint32_t r = rb->read;
    *byte = rb->data[r];
    rb->read = (r + 1) & BUFFER_MASK;
    return true;
}

/**
 * @brief 查询已用空间大小（不加锁，调用者保证安全）
 */
uint32_t rb_used_space(const RingBuffer *rb) {
    uint32_t w = rb->write;
    uint32_t r = rb->read;
    if (w >= r) {
        return w - r;
    } else {
        return BUFFER_SIZE - (r - w);
    }
}

/**
 * @brief 查询剩余空闲空间
 */
uint32_t rb_free_space(RingBuffer *rb) {
    return BUFFER_SIZE - 1 - rb_used_space(rb);
}

/**
 * @brief 批量写入字节序列，返回实际写入的字节数
 */
uint32_t rb_write_bytes(RingBuffer *rb, const uint8_t *data, uint32_t len) {
    if (len == 0 || rb_is_full(rb)) {
        return 0;
    }

    uint32_t free_space = rb_free_space(rb);
    uint32_t write_len = MY_MIN(len, free_space);
    uint32_t w = rb->write;

    // 第一段可写长度（从 w 到缓冲区末尾）
    uint32_t first_part = MY_MIN(BUFFER_SIZE - w, write_len);
    memcpy(&rb->data[w], data, first_part);

    // 可能存在的第二段（从头开始）
    uint32_t second_part = write_len - first_part;
    if (second_part) {
        memcpy(rb->data, data + first_part, second_part);
    }

    // 更新写指针（利用掩码自动回绕）
    rb->write = (w + write_len) & BUFFER_MASK;
    return write_len;
}

/**
 * @brief 写入字符串（不含 '\0'）
 */
uint32_t rb_write_string(RingBuffer *rb, const char *str) {
    if (str == NULL) return 0;
    size_t len = strlen(str);
    if (len > UINT32_MAX) len = UINT32_MAX;
    return rb_write_bytes(rb, (const uint8_t *)str, (uint32_t)len);
}

/**
 * @brief 格式化写入（使用固定缓冲区，避免大栈分配）
 */
uint32_t rb_printf(RingBuffer *rb, const char *format, ...) {
    if (rb == NULL || format == NULL) return 0;

    char buf[128];   // 固定大小，性能安全
    va_list args;
    va_start(args, format);
    int len = vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    if (len < 0) return 0;
    if (len >= (int)sizeof(buf)) {
        len = sizeof(buf) - 1;      // 被截断，但缓冲仍以 '\0' 结尾
    }

    return rb_write_bytes(rb, (const uint8_t *)buf, (uint32_t)len);
}

/**
 * @brief 查看数据但不消费，返回拷贝的字节数
 */
uint32_t rb_peek_bytes(const RingBuffer *rb, uint8_t *dest, uint32_t max_len) {
    if (max_len == 0 || rb_is_empty(rb)) {
        return 0;
    }

    uint32_t used = rb_used_space(rb);
    uint32_t copy_len = MY_MIN(used, max_len);
    uint32_t r = rb->read;

    uint32_t first_part = MY_MIN(BUFFER_SIZE - r, copy_len);
    memcpy(dest, &rb->data[r], first_part);

    if (copy_len > first_part) {
        memcpy(dest + first_part, rb->data, copy_len - first_part);
    }

    return copy_len;
}

/**
 * @brief 读取数据并消费（移动读指针）
 */
uint32_t rb_read_bytes(RingBuffer *rb, uint8_t *dest, uint32_t max_len) {
    uint32_t copied = rb_peek_bytes(rb, dest, max_len);
    if (copied > 0) {
        rb->read = (rb->read + copied) & BUFFER_MASK;
    }
    return copied;
}

/**
 * @brief 丢弃指定长度数据
 */
uint32_t rb_skip_bytes(RingBuffer *rb, uint32_t len) {
    uint32_t used = rb_used_space(rb);
    uint32_t skip_len = MY_MIN(len, used);
    rb->read = (rb->read + skip_len) & BUFFER_MASK;
    return skip_len;
}

/**
 * @brief 读取所有可用数据（消费），合并逻辑减少调用层
 */
uint32_t rb_read_all(RingBuffer *rb, uint8_t *dest, uint32_t max_len) {
    if (max_len == 0 || rb_is_empty(rb)) {
        return 0;
    }

    uint32_t used = rb_used_space(rb);
    uint32_t read_len = MY_MIN(used, max_len);
    uint32_t r = rb->read;

    // 直接拷贝，避免再进入 rb_peek_bytes 和 rb_read_bytes 的二次判断
    uint32_t first_part = MY_MIN(BUFFER_SIZE - r, read_len);
    memcpy(dest, &rb->data[r], first_part);
    if (read_len > first_part) {
        memcpy(dest + first_part, rb->data, read_len - first_part);
    }

    rb->read = (r + read_len) & BUFFER_MASK;
    return read_len;
}