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
#include <unordered_map>

typedef struct DocEntry_ {
    uint32 docid;
    uint64 idx;
    uint32 num_words;
    uint32 num_common_words;
} DocEntry;

typedef struct WordEntry_ {
    uint32 wordid;
    int topicid;
} WordEntry;

typedef struct TopicEntry_ {
    int topicid;
    uint64 num_words;
    std::unordered_map<uint32, uint64> *word_lookup;
} TopicEntry;

void topicEntryInit(TopicEntry *topic_entry, int topicid);
void topicEntryDestory(TopicEntry *topic_entry);
int getTopicWordCnt(TopicEntry *topic_entry, uint32 wordid);
void setTopicWordCnt(TopicEntry *topic_entry, uint32 wordid, uint64 newval);

#endif //MODEL_H
