/* ========================================================
 *   Copyright (C) 2019 All rights reserved.
 *   
 *   filename : model.h
 *   author   : ***
 *   date     : 2019-12-06
 *   info     : 
 * ======================================================== */
#ifndef _MODEL_H
#define _MODEL_H

#include "utils.h"

typedef struct DocEntry_ {
    uint32 docid;
    uint32 idx;
    uint32 num_words;
    uint32 *topic_dist; // size is num_topic + 1
} DocEntry;

void docEntryInit(DocEntry *doc_entry, uint32 docid, uint32 num_topics);
void docEntryDestory(DocEntry *doc_entry);
uint32 getDocTopicCnt(DocEntry *doc_entry, int topicid);
void addDocTopicCnt(DocEntry *doc_entry, int topicid, int delta);

typedef struct WordEntry_ {
    uint32 wordid;
    int topicid;
} WordEntry;

typedef struct TopicEntry_ {
    int topicid;
    uint32 num_words;
    uint32 *word_dist; // size is vocab_size
} TopicEntry;

void topicEntryInit(TopicEntry *topic_entry, int topicid, uint32 vocab_size);
void topicEntryDestory(TopicEntry *topic_entry);
uint32 getTopicWordCnt(TopicEntry *topic_entry, uint32 wordid);
void addTopicWordCnt(TopicEntry *topic_entry, uint32 wordid, int delta);

#endif //MODEL_H
