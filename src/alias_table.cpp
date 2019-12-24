#include "alias_table.h"
#include <cstdlib>
#include <cstdio>
#include <vector>

void AliasTableInit(AliasTable *table, uint32 wordid, int num_topics) {
    table->wordid = wordid;
    table->num_topics = num_topics;
    table->num_sampled = 0;
    table->sum_dist = 0.;
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
    std::vector<int> A, B;

    // normalize dist
    table->sum_dist = 0.;
    for (t = 0; t < table->num_topics; t++) {
        table->sum_dist += dist[t];
    }
    for (t = 0; t < table->num_topics; t++) {
        table->P[t] = dist[t] / table->sum_dist * num_topics;
        if (table->P[t] > 1.) A.push_back(t);
        else if (table->P[t] < 1.) B.push_back(t);
        table->P[t] = table->P[t];
    }
    // construct alias table
    while (B.size() > 0) {
        ta = A[A.size()];
        tb = B[B.size()];
        A.pop_back();
        B.pop_back();
        table->P[ta] -= 1 - table->P[tb];
        table->G[tb] = ta;
        if (table->P[ta] > 1.) A.push_back(ta);
        else if (table->P[ta] < 1.) B.push_back(ta);
    }
    // reset num_sampled
    table->num_sampled = 0;
}

int sampleAliasTable(AliasTable *table) {
    int t;

    t = rand() % table->num_topics;
    if ((real)rand() / (real)(RAND_MAX + 1) < table->P[t]) return t;
    else return table->Q[t];
}
