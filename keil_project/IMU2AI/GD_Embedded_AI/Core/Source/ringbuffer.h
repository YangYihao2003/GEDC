#ifndef RINGBUFFER_H
#define RINGBUFFER_H
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

#define BUFFER_SIZE 2048  // 实际容量为 1023
#define BUFFER_MASK  (BUFFER_SIZE - 1)

typedef struct {
    uint8_t data[BUFFER_SIZE];
    volatile uint32_t write; // 写索引
    volatile uint32_t read;  // 读索引
		SemaphoreHandle_t rb_mutex;  // 互斥锁
} RingBuffer;

// 内联小函数
static inline bool rb_is_empty(const RingBuffer *rb) {
    return rb->write == rb->read;
}

static inline bool rb_is_full(const RingBuffer *rb) {
    return ((rb->write + 1) & BUFFER_MASK) == rb->read;
}


void rb_init(RingBuffer *rb);

bool rb_put(RingBuffer *rb, uint8_t byte);
bool rb_get(RingBuffer *rb, uint8_t *byte);
uint32_t rb_used_space(const RingBuffer *rb);
uint32_t rb_write_string(RingBuffer *rb, const char *str);
uint32_t rb_write_bytes(RingBuffer *rb, const uint8_t *data, uint32_t len);
uint32_t rb_free_space(RingBuffer *rb);
uint32_t rb_peek_bytes(const RingBuffer *rb, uint8_t *dest, uint32_t max_len);
uint32_t rb_read_bytes(RingBuffer *rb, uint8_t *dest, uint32_t max_len);
uint32_t rb_skip_bytes(RingBuffer *rb, uint32_t len);  // 直接跳过/丢弃数据
uint32_t rb_read_all(RingBuffer *rb, uint8_t *dest, uint32_t max_len);
uint32_t rb_printf(RingBuffer *rb, const char *format, ...);
#endif