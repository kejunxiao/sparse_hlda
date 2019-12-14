#include "dlist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dlist_init(DList *list, void (*destory)(void *data)) {
    list->size = 0;
    list->destory = destory;
    list->head = NULL;
    list->tail = NULL;
}

void dlist_destory(DList *list) {
    void *data = NULL;
    while (dlist_size(list) > 0) {
        if ((0 == dlist_remove(list, dlist_head(list), (void **)&data)) && NULL != list->destory) {
            list->destory(data);
        }
    }
    memset((void *)list, 0, sizeof(DList));
}

int dlist_ins_next(DList *list, DListElmt *element, const void *data) {
    if (element == NULL && dlist_size(list) != 0) {
        return -1;
    }
    DListElmt *new_element = (DListElmt *)malloc(sizeof(DListElmt));
    new_element->data = (void *)data;
    if (dlist_size(list) == 0) {
        list->head = new_element;
        list->tail = new_element;
        new_element->next = NULL;
        new_element->prev = NULL;
    } else {
        element->next = new_element;
        new_element->prev = element;
        new_element->next = element->next;
        if (dlist_tail(list) == element) {
            list->tail = new_element;
        } else {
            element->next->prev = new_element;
        }
    }
    list->size++;
    return 0;
}

int dlist_ins_prev(DList *list, DListElmt *element, const void *data) {
    if (element == NULL && dlist_size(list) != 0) {
        return -1;
    }
    DListElmt *new_element = (DListElmt *)malloc(sizeof(DListElmt));
    new_element->data = (void *)data;
    if (dlist_size(list) == 0) {
        list->head = new_element;
        list->tail = new_element;
        new_element->next = NULL;
        new_element->prev = NULL;
    } else {
        element->prev = new_element;
        new_element->next = element;
        new_element->prev = element->prev;
        if (dlist_head(list) == element) {
            list->head = new_element;
        } else {
            element->prev->next = new_element;
        }
    }
    list->size++;
    return 0;
}

int dlist_remove(DList *list, DListElmt *element, void **data) {
    if (element == NULL || dlist_size(list) == 0) {
        return -1;
    }

    if (dlist_head(list) == element) { // element is head
        list->head = element->next;
    } else {
        element->prev->next = element->next;
    }

    if (dlist_tail(list) == element) { // element is tail
        list->tail = element->prev;
    } else {
        element->next->prev = element->prev;
    }
    *data = element->data;
    free(element);
    list->size--;
    return 0;
}
