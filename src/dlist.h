#ifndef _DLIST_H
#define _DLIST_H

typedef struct DListElmt_ {
    void *data;
    struct DListElmt_ *next;
    struct DListElmt_ *prev;
} DListElmt;

typedef struct DList_ {
    int size;
    int (*match)(const void *key1, const void *key2);
    void (*destory)(void *data);
    DListElmt *head;
    DListElmt *tail;
} DList;

/* public interface */
void dlist_init(DList *list, void (*destroy)(void *data));
void dlist_destory(DList *list);
int dlist_ins_next(DList *list, DListElmt *element, const void *data);
int dlist_ins_prev(DList *list, DListElmt *element, const void *data);
int dlist_remove(DList *list, DListElmt *element, void **data);

#define dlist_size(list) ((list)->size)
#define dlist_head(list) ((list)->head)
#define dlist_tail(list) ((list)->tail)
#define dlist_is_head(list, element) ((element) == (list)->head ? 1 : 0)
#define dlist_is_tail(list, element) ((element) == (list)->tail ? 1 : 0)
#define dlist_data(element) ((element)->data)
#define dlist_next(element) ((element)->next)
#define dlist_prev(element) ((element)->prev)

#endif //DLIST_H
