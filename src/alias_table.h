#include "util.h"

typedef struct AliasTable_ {
    uint32 wordid;
    uint32 num_topics;
    uint32 num_sampled;
    real Q_w;
    real *wbucket;
    real *P; // probs bucket, index is topicid, value is prob
    int *G;  // other topicid bucket
} AliasTable;

void aliasTableInit(AliasTable *table, uint32 wordid, uint32 num_topics);
void aliasTableDestory(AliasTable *table);
void generateAliasTable(AliasTable *table, real *wbucket, real Q_w);
int sampleAliasTable(AliasTable *table);
