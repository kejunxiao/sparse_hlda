#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

void topicNodeInit(TopicNode *topic_node, int topicid) {
    topic_node->prev = NULL;
    topic_node->next = NULL;
    topic_node->cnt = 0;
    topic_node->topicid = topicid;
}

void docEntryInit(DocEntry *doc_entry, uint32 docid) {
    doc_entry->docid = docid;
    doc_entry->idx = 0;
    doc_entry->num_words = 0;
    doc_entry->nonzeros = NULL;
}

void wordEntryInit(WordEntry *word_entry, uint32 wordid) {
    word_entry->wordid = wordid;
    word_entry->nonzeros = NULL;
}

uint32 getDocTopicCnt(TopicNode *doc_topic_dist, uint32 num_topics, uint32 docid, int topicid) {
    return doc_topic_dist[docid * (1 + num_topics) + topicid].cnt;
}

void addDocTopicCnt(TopicNode *doc_topic_dist, uint32 num_topics, DocEntry *doc_entry, int topicid, int delta) {
    uint32 oldcnt, offset;
    TopicNode *node;

    offset = doc_entry->docid * (1 + num_topics) + topicid;
    oldcnt = doc_topic_dist[offset].cnt;
    doc_topic_dist[offset].cnt += delta;

    if (topicid == num_topics) return; // no insert common-topicid 
    if (oldcnt == 0 && delta > 0) { 
        // insert topicid into nonzeros of docid
        node = &doc_topic_dist[offset];
        node->next = doc_entry->nonzeros;
        if (doc_entry->nonzeros) (doc_entry->nonzeros)->prev = node;
        doc_entry->nonzeros = node;
    } else if (doc_topic_dist[offset].cnt == 0 && delta < 0) {
        // remove topicid from nonzeros of docid
        node = &doc_topic_dist[offset];
        if (node->prev) node->prev->next = node->next;
        else doc_entry->nonzeros = node->next;
        if (node->next) node->next->prev = node->prev;
        node->prev = NULL;
        node->next = NULL;
    }
}

uint32 getTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, uint32 wordid) {
    return topic_word_dist[wordid * (1 + num_topics) + topicid].cnt;
}

void addTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, WordEntry *word_entry, int delta) {
    uint32 oldcnt, offset;
    TopicNode *node;

    offset = word_entry->wordid * (1 + num_topics) + topicid;
    oldcnt = topic_word_dist[offset].cnt;
    topic_word_dist[offset].cnt += delta;

    if (topicid == num_topics) return; // no insert common-topicid
    if (oldcnt == 0 && delta > 0) { 
        // insert topicid into nozeros of wordid
        node = &topic_word_dist[offset];
        node->next = word_entry->nonzeros;
        if (word_entry->nonzeros) (word_entry->nonzeros)->prev = node;
        word_entry->nonzeros = node;
    } else if (topic_word_dist[offset].cnt == 0 && delta < 0) {
        // remove topicid from nonzeros of wordid
        node = &topic_word_dist[offset];
        if (node->prev) node->prev->next = node->next;
        else word_entry->nonzeros = node->next;
        if (node->next) node->next->prev = node->prev;
        node->prev = NULL;
        node->next = NULL;
    }
}
