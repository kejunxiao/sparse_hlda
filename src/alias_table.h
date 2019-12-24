#include "util.h"

typedef struct AliasTable_ {
    uint32 wordid;
    int num_topics;
    real *P;
    int *G;
} AliasTable;

void AliasTableInit(AliasTable *table, uint32 wordid, int num_topics);
void AliasTableDestory(AliasTable *table);
void generateAliasTable(AliasTable *table, real *dist);
int sampleAliasTable(AliasTable *table);
