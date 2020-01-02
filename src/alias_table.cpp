#include "alias_table.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

void aliasTableInit(AliasTable *table, uint32 wordid, uint32 num_topics) {
    table->wordid = wordid;
    table->num_topics = num_topics;
    table->num_sampled = 0;
    table->Q_w = 0;
    if (NULL == (table->wbucket = (real *)calloc(num_topics, sizeof(real)))) {
        fprintf(stderr, "ERROR: allocate memory for AliasTable fail\n");
        exit(1);
    }
    memset(table->wbucket, 0, sizeof(real) * num_topics);
    if (NULL == (table->P = (real *)calloc(num_topics, sizeof(real)))) {
        fprintf(stderr, "ERROR: allocate memory for AliasTable fail\n");
        exit(1);
    }
    memset(table->P, 0, sizeof(real) * num_topics);
    if (NULL == (table->G = (int *)calloc(num_topics, sizeof(int)))) {
        fprintf(stderr, "ERROR: allocate memory for AliasTable fail\n");
        exit(1);
    }
    memset(table->G, 0, sizeof(int) * num_topics);
}

void aliasTableDestory(AliasTable *table) {
    free(table->P);
    free(table->G);
    table->P = NULL;
    table->G = NULL;
}

void generateAliasTable(AliasTable *table, real *wbucket, real Q_w) {
    int t, ta, tb;
    std::vector<int> A, B;

    // normalize wbucket
    table->Q_w = Q_w;
    memcpy(table->wbucket, wbucket, table->num_topics * sizeof(real));
    for (t = 0; t < table->num_topics; t++) {
        table->P[t] = wbucket[t] / Q_w * table->num_topics;
        if (table->P[t] > 1) A.push_back(t);
        else if (table->P[t] < 1) B.push_back(t);
    }
    // construct alias table
    while (!A.empty() && !B.empty()) {
        ta = A.back();
        tb = B.back();
        A.pop_back();
        B.pop_back();
        table->P[ta] -= 1 - table->P[tb];
        table->G[tb] = ta;
        if (table->P[ta] > 1) A.push_back(ta);
        else if (table->P[ta] < 1) B.push_back(ta);
    }
    // reset num_sampled
    table->num_sampled = 0;
    #ifdef DEBUG
    fprintf(stderr, "wordid = %d, alias table is: ", table->wordid);
    for (int x = 0; x < table->num_topics; x++) {
        fprintf(stderr, "%d:%.4f:%d ", x, table->P[x], table->G[x]);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
    #endif
}

int sampleAliasTable(AliasTable *table) {
    int t;

    t = rand() % table->num_topics;
    if ((real)rand() / (RAND_MAX + 1.) < table->P[t]) return t;
    else return table->G[t];
}
