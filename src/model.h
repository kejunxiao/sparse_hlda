#ifndef _MODEL_H
#define _MODEL_H

#include "utils.h"

typedef struct TopicNode_ {
    TopicNode_ *prev;
    TopicNode_ *next;
    uint32 cnt;
    int topicid;
} TopicNode;
    
typedef struct DocEntry_ {
    uint32 docid;
    uint32 idx;
    uint32 num_words;
    TopicNode *nonzeros;
} DocEntry;

typedef struct WordEntry_ {
    uint32 wordid;
    TopicNode *nonzeros;
} WordEntry;

typedef struct TokenEntry_ {
    uint32 wordid;
    int topicid;
} TokenEntry;

/* public interface */
void topicNodeInit(TopicNode *topic_node, int topicid);
void docEntryInit(DocEntry *doc_entry, uint32 docid);
void wordEntryInit(WordEntry *word_entry, uint32 wordid);

inline uint32 getDocTopicCnt(TopicNode *doc_topic_dist, uint32 num_topics, uint32 docid, int topicid) {
    return doc_topic_dist[docid * (1 + num_topics) + topicid].cnt;
}

void addDocTopicCnt(TopicNode *doc_topic_dist, uint32 num_topics, DocEntry *doc_entry, int topicid, int delta);

inline uint32 getTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, uint32 wordid) {
    return topic_word_dist[wordid * (1 + num_topics) + topicid].cnt;
}

void addTopicWordCnt(TopicNode *topic_word_dist, uint32 num_topics, int topicid, WordEntry *word_entry, int delta);

#endif //MODEL_H
