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

void docEntryInit(DocEntry *doc_entry, uint32 docid, uint32 num_topics) {
    doc_entry->docid = docid;
    doc_entry->idx = 0;
    doc_entry->num_words = 0;
    if (NULL == (doc_entry->topic_dist = (uint32 *)calloc(num_topics + 1, sizeof(uint32)))) {
        fprintf(stderr, "ERROR: allocate memory for doc-topic distribution fail\n");
        exit(1);
    }
    memset(doc_entry->topic_dist, 0, (num_topics + 1) * sizeof(uint32));
}

void docEntryDestory(DocEntry *doc_entry) {
    free(doc_entry->topic_dist);
}

uint32 getDocTopicCnt(DocEntry *doc_entry, int topicid) {
    return doc_entry->topic_dist[topicid];
}

void addDocTopicCnt(DocEntry *doc_entry, int topicid, int delta) {
    if ((long long)doc_entry->topic_dist[topicid] < -delta ) {
        fprintf(stderr, "ERROR: after modeified (delta = %d), topic %d (count = %d) in doc %d < 0\n", delta, topicid, doc_entry->topic_dist[topicid], doc_entry->docid);
        exit(1);
    }
    doc_entry->topic_dist[topicid] += delta;
}

void topicEntryInit(TopicEntry *topic_entry, int topicid, uint32 vocab_size) {
    topic_entry->topicid = topicid;
    topic_entry->num_words = 0;
    if (NULL == (topic_entry->word_dist = (uint32 *)calloc(vocab_size, sizeof(uint32)))) {
        fprintf(stderr, "ERROR: allocate memory for topic-word distribution fail\n");
        exit(1);
    }
    memset(topic_entry->word_dist, 0, vocab_size * sizeof(uint32));
}

void topicEntryDestory(TopicEntry *topic_entry) {
    free(topic_entry->word_dist);
}

uint32 getTopicWordCnt(TopicEntry *topic_entry, uint32 wordid) {
    return topic_entry->word_dist[wordid];
}

void addTopicWordCnt(TopicEntry *topic_entry, uint32 wordid, int delta) {
    if ((long long)topic_entry->word_dist[wordid] < -delta) {
        fprintf(stderr, "ERROR: after modeified (delta = %d), word %d (count = %d) in topic %d < 0\n", delta, wordid, topic_entry->word_dist[wordid], topic_entry->topicid);
        exit(1);
    }
    topic_entry->word_dist[wordid] += delta;
    topic_entry->num_words = (long long)topic_entry->num_words + delta;
}
