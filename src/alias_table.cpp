#include "alias_table.h"
#include <cstdlib>
#include <cstdio>
#include <vector>

void AliasTableInit(AliasTable *table, uint32 wordid, int num_topics) {
    table->wordid = wordid;
    table->num_topics = num_topics;
    if (NULL == (table->P = (real *)calloc(num_topics, sizeof(real)))) {
        fprintf(stderr, "ERROR: allocate memory for AliasTable fail\n");
        exit(1);
    }
    memset(table->P, 0, sizeof(real) * num_topics);
    if (NULL == (table->Q = (int *)calloc(num_topics, sizeof(int)))) {
        fprintf(stderr, "ERROR: allocate memory for AliasTable fail\n");
        exit(1);
    }
    memset(table->G, 0, sizeof(int) * num_topics);
}

void AliasTableDestory(AliasTable *table) {
    free(table->P);
    free(table->Q);
    table->P = NULL;
    table->Q = NULL;
}

void generateAliasTable(AliasTable *table, real *dist) {
    int t, ta, tb;
    real sum, tmp;
    std::vector<int> A, B;

    // normalize dist
    sum = 0.;
    for (t = 0; t < table->num_topics; t++) {
        sum += dist[t];
    }
    for (t = 0; t < table->num_topics; t++) {
        tmp = dist[t] / sum * num_topics;
        if (tmp > 1.) A.push_back(t);
        else if (tmp < 1.) B.push_back(t);
        table->P[t] = tmp;
    }
    // construct alias table
    while (B.size() > 0) {
        ta = A.pop_back();
        tb = B.pop_back();
        table->P[ta] -= 1 - table->P[tb];
        table->G[tb] = ta;
        if (table->P[ta] > 1.) A.push_back(ta);
        else if (table->P[ta] < 1.) B.push_back(ta);
    }
}

int sampleAliasTable(AliasTable *table) {
    int t;

    t = rand() % table->num_topics;
    if ((real)rand() / (real)(RAND_MAX + 1) < table->P[t]) return t;
    else return table->Q[t];
}
