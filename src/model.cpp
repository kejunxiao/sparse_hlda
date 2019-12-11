/* ========================================================
 *   Copyright (C) 2019 All rights reserved.
 *   
 *   filename : model.cpp
 *   author   : ***
 *   date     : 2019-12-06
 *   info     : 
 * ======================================================== */
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

void topicEntryInit(TopicEntry *topic_entry, int topicid) {
    topic_entry->topicid = topicid;
    topic_entry->num_words = 0;
    if (NULL == (topic_entry->word_lookup = new std::unordered_map<uint32, uint64>())) {
        fprintf(stderr, "ERROR: create hash table for topic_entry fail");
        exit(1);
    }
}

void topicEntryDestory(TopicEntry *topic_entry) {
    topic_entry->num_words = 0;
    delete topic_entry->word_lookup;
}

int getTopicWordCnt(TopicEntry *topic_entry, uint32 wordid) {
    std::unordered_map<uint32, uint64>::iterator itr = topic_entry->word_lookup->find(wordid);
    if (itr != topic_entry->word_lookup->end()) return itr->second;
    else return 0;
}

void setTopicWordCnt(TopicEntry *topic_entry, uint32 wordid, uint64 newval) {
    (*(topic_entry->word_lookup))[wordid] = newval;
}
