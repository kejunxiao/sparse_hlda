#include "utils.h"
#include "model.h"
#include "alias_table.h"
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include <string>

/* gloabl variables */
// parameters
char input[MAX_STRING];
char output[MAX_STRING];
uint32 num_topics = 0;
real alpha = 0.05;
real beta = 0.01;
real gamma0 = 0.1;
real beta2 = 0.01;
uint32 num_iters = 20;
int save_step = -1;
uint32 MH_step = 2;

// train data related
uint32 num_docs = 0;
uint32 vocab_size = 0;
uint32 num_tokens = 0;
std::unordered_map<std::string, uint32> word2id;
std::unordered_map<uint32, std::string> id2word;

// model related
uint32 *topic_row_sums = NULL;
TopicNode *doc_topic_dist = NULL;
uint32 *topic_word_dist = NULL;

DocEntry *doc_entries = NULL;
AliasTable *alias_tables = NULL;
TokenEntry *token_entries = NULL;

/* helper functions */
static void getWordFromId(uint32 wordid, char *word) {
    std::unordered_map<uint32, std::string>::iterator itr = id2word.find(wordid);
    if (itr != id2word.end()) {
        strcpy(word, itr->second.c_str());
        return;
    } else { 
        fprintf(stderr, "***ERROR***: unknown wordid %d", wordid); 
        exit(1);
    }
}

static uint32 getIdFromWord(const char *word) {
    std::string s(word);
    std::unordered_map<std::string, uint32>::iterator itr = word2id.find(s);
    if (itr != word2id.end()) {
        return itr->second;
    } else { 
        word2id[s] = vocab_size;
        id2word[vocab_size] = s;
        vocab_size++;
        return vocab_size;
    }
}

inline static int genRandTopicId() { return rand() % num_topics; }

/* LightLDA */
// topic-word cnt
inline void addTopicWordCntAlias(uint32 *topic_word_dist, uint32 num_topics, int topicid, uint32 wordid, int delta) {
    topic_word_dist[wordid * (1 + num_topics) + topicid] += delta;
}

inline uint32 getTopicWordCntAlias(uint32 *topic_word_dist, uint32 num_topics, int topicid, uint32 wordid) {
    return topic_word_dist[wordid * (1 + num_topics) + topicid];
}

// denominators
static void initDenomin(real *denominators, real Vbeta) {
    int t;
    for (t = 0; t < num_topics; t++) denominators[t] = Vbeta + topic_row_sums[t];
}

inline static void updateDenomin(real *denominators, real Vbeta, int topicid) {
    denominators[topicid] = Vbeta + topic_row_sums[topicid];
}

// doc_topic & topic_word bucket
static real initDW(real *dwbucket, DocEntry *doc_entry, uint32 wordid, real *denominators) {
    TopicNode *node;
    real P_dw = 0;

    node = doc_entry->nonzeros;
    while (node) {
        dwbucket[node->topicid] = node->cnt * (getTopicWordCntAlias(topic_word_dist, num_topics, node->topicid, wordid) + beta) / denominators[node->topicid]; 
        P_dw += dwbucket[node->topicid];
        node = node->next;
    }
    return P_dw;
}

// topic-word bucket
static real initW(real *wbucket, uint32 wordid, real *denominators) {
    int t;
    real Q_w = 0;

    for (t = 0; t < num_topics; t++) {
        wbucket[t] = alpha * (getTopicWordCntAlias(topic_word_dist, num_topics, t, wordid) + beta) / denominators[t];
        Q_w += wbucket[t];
    }

    return Q_w;
}

// common-word bucket
inline static real initComm(real Vbeta2, uint32 wordid) {
    return (getTopicWordCntAlias(topic_word_dist, num_topics, num_topics, wordid) + beta2) / (topic_row_sums[num_topics] + Vbeta2);
}


/* public interface */
void learnVocabFromDocs() {
    uint32 a, len;
    char ch, *token, buf[MAX_STRING];
    FILE *fin;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "***ERROR***: can not open input file");
        exit(1);
    }
    // get number of documents and number of tokens from input file
    len = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            token = strtok(buf, ":");  // get word-string
            getIdFromWord(token);
            token = strtok(NULL, ":"); // get word-freq
            num_tokens += atoi(token);
            memset(buf, 0, len);
            len = 0;
            if (ch == '\n') {
                num_docs++;
                if (num_docs % 1000 == 0) {
                    printf("%dK%c", num_docs / 1000, 13);
                    fflush(stdout);
                }
            }
        } else { // append ch to buf
            buf[len] = ch;
            len++;
        }
    }
    printf("number of documents: %d, number of tokens: %d, vocabulary size: %d\n", num_docs, num_tokens, vocab_size);

    // allocate memory for doc-topic distribution
    doc_topic_dist = (TopicNode *)calloc(num_docs * (1 + num_topics), sizeof(TopicNode));
    for (a = 0; a < num_docs * (1 + num_topics); a++) topicNodeInit(&doc_topic_dist[a], a % (1 + num_topics));
    // allocate memory for topic-word distribution
    topic_word_dist = (uint32 *)calloc(vocab_size * (1 + num_topics), sizeof(uint32));
    memset(topic_word_dist, 0, sizeof(uint32) * vocab_size * (1 + num_topics));
    // allocate memory for doc_entries
    doc_entries = (DocEntry *)calloc(num_docs, sizeof(DocEntry));
    for (a = 0; a < num_docs; a++) docEntryInit(&doc_entries[a], a);
    // allocate memory for alias table for every word
    alias_tables = (AliasTable *)calloc(vocab_size, sizeof(AliasTable));
    for (a = 0; a < vocab_size; a++) aliasTableInit(&alias_tables[a], a, num_topics);
    // allocate memory for token_entries
    token_entries = (TokenEntry *)calloc(num_tokens, sizeof(TokenEntry));
}

void loadDocs() {
    int topicid;
    uint32 a, b, c, freq, len, wordid, docid;
    char ch, buf[MAX_STRING], *token;
    FILE *fin;
    DocEntry *doc_entry;
    TokenEntry *token_entry;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "***ERROR***: can not open input file");
        exit(1);
    }
    // load documents
    docid = 0;
    len = 0;
    b = 0;
    c = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\n') {
            buf[len] = '\0';
            token = strtok(buf, ":");  // get word-string
            wordid = getIdFromWord(token);  
            token = strtok(NULL, ":"); // get word-freq
            freq = atoi(token);
            token = strtok(NULL, ":"); // get anchor-topic
            topicid = atoi(token);

            doc_entry = &doc_entries[docid];
            for (a = 0; a < freq; a++) {
                token_entry = &token_entries[c + a];
                token_entry->wordid = wordid;
                if (topicid >= 0) token_entry->topicid = topicid;
                else token_entry->topicid = genRandTopicId();
                addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, 1);
                addTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, wordid, 1);
                topic_row_sums[token_entry->topicid]++;
            }
            c += freq;
            memset(buf, 0, len);
            len = 0;
            if (ch == '\n') {
                doc_entry = &doc_entries[docid];
                doc_entry->idx = b;
                doc_entry->num_words = c - b;

                docid++;
                b = c;
                if (docid % 1000 == 0) {
                    printf("%dK%c", docid / 1000, 13);
                    fflush(stdout);
                }
            }
        } else { // append ch to buf
            if (len >= MAX_STRING) continue;
            buf[len] = ch;
            len++;
        }
    }
}

void initTable() {
    uint32 wordid;
    real Q_w, *denominators, *wbucket;
    real Vbeta = vocab_size * beta;
    
    denominators = (real *)calloc(num_topics, sizeof(real));
    wbucket = (real *)calloc(num_topics, sizeof(real));

    memset(denominators, 0, num_topics * sizeof(real));
    initDenomin(denominators, Vbeta);

    for (wordid = 0; wordid < vocab_size; wordid++) {
        memset(wbucket, 0, num_topics * sizeof(real));
        Q_w = initW(wbucket, wordid, denominators);
        generateAliasTable(&alias_tables[wordid], wbucket, Q_w);
    }

    free(denominators);
    free(wbucket);
}

void MHSample(uint32 round) {
    uint32 a, b, c;
    int new_topicid;
    struct timeval tv1, tv2;
    real P_dw, Q_w, spec_topic_r, s_spec, s_comm, r, s, accept, *denominators, *dwbucket, *wbucket;
    real Kalpha = num_topics * alpha, Vbeta = vocab_size * beta, Vbeta2 = vocab_size * beta2;
    DocEntry *doc_entry;
    TokenEntry *token_entry;
    AliasTable *table;
    TopicNode *node;

    denominators = (real *)calloc(num_topics, sizeof(real));
    dwbucket = (real *)calloc(num_topics, sizeof(real));
    wbucket = (real *)calloc(num_topics, sizeof(real));

    memset(denominators, 0, num_topics * sizeof(real));
    initDenomin(denominators, Vbeta);

    gettimeofday(&tv1, NULL);
    for (a = 0; a < num_docs; a++) {
        if (a > 0 && a % 10000 == 0) {
            gettimeofday(&tv2, NULL);
            printf("%cProcess: %.2f%% Documents/Sec: %.2fK",
                   13,
                   (round + a * 1. / num_docs) * 100. / num_iters,
                   10. / (tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1000000.));
            fflush(stdout);
            memcpy(&tv1, &tv2, sizeof(struct timeval));
        }
        doc_entry = &doc_entries[a];

        for (b = 0; b < doc_entry->num_words; b++) {
            token_entry = &token_entries[doc_entry->idx + b];
            table = &alias_tables[token_entry->wordid];

            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, -1);
            addTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid, -1);
            doc_entry->num_words--;
            topic_row_sums[token_entry->topicid]--;
            if (token_entry->topicid < num_topics) {
                updateDenomin(denominators, Vbeta, token_entry->topicid);
            }

            memset(dwbucket, 0, num_topics * sizeof(real));
            P_dw = initDW(dwbucket, doc_entry, token_entry->wordid, denominators);

            spec_topic_r = (gamma0 + doc_entry->num_words - getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics)) / (1 + doc_entry->num_words);

            s_spec = (P_dw + table->Q_w) * spec_topic_r / (Kalpha + doc_entry->num_words - getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics));
            s_comm = (1. - spec_topic_r) * initComm(Vbeta2, token_entry->wordid);
            r = (s_spec + s_comm) * rand() / RAND_MAX;
            // start sampling
            new_topicid = -1;
            s = 0;
            if (r < s_spec) {
                // sample in special topics, topicid range 0 ~ num_topics - 1
                for (c = 0; c < MH_step; c++) {
                    r = (P_dw + table->Q_w) * rand() / (RAND_MAX + 1.);
                    if (r < P_dw) {
                        node = doc_entry->nonzeros;
                        while (node) {
                            s += dwbucket[node->topicid];
                            if (s > r) {new_topicid = node->topicid; break;}
                            node = node->next;
                        }
                    } else {
                        table->num_sampled++;
                        if (table->num_sampled > num_topics / 2) {
                            memset(wbucket, 0, num_topics * sizeof(real));
                            addTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid, 1);
                            topic_row_sums[token_entry->topicid]++;
                            if (token_entry->topicid < num_topics) {
                                updateDenomin(denominators, Vbeta, token_entry->topicid);
                            }
                            Q_w = initW(wbucket, token_entry->wordid, denominators);
                            generateAliasTable(table, wbucket, Q_w);
                            addTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid, -1);
                            topic_row_sums[token_entry->topicid]--;
                            if (token_entry->topicid < num_topics) {
                                updateDenomin(denominators, Vbeta, token_entry->topicid);
                            }
                        }
                        new_topicid = sampleAliasTable(table);
                    }
                    // acceptance
                    if ((token_entry->topicid < num_topics) && (new_topicid != token_entry->topicid)) {
                        accept = (getDocTopicCnt(doc_topic_dist, num_topics, a, new_topicid) + alpha) /
                            (getDocTopicCnt(doc_topic_dist, num_topics, a, token_entry->topicid) + alpha);
                        accept *= (getTopicWordCntAlias(topic_word_dist, num_topics, new_topicid, token_entry->wordid) + beta) / 
                            (getTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid) + beta);
                        accept *= (topic_row_sums[token_entry->topicid] + Vbeta) /
                            (topic_row_sums[new_topicid] + Vbeta);
                        accept *= (dwbucket[token_entry->topicid] + table->wbucket[token_entry->topicid]) /
                            (dwbucket[new_topicid] + table->wbucket[new_topicid]);
                        if ((real)rand() / (RAND_MAX + 1.) < accept) token_entry->topicid = new_topicid;
                        #ifdef DEBUG
                        fprintf(stderr, "wordid = %d, aceept = %.16f, new_topic = %d, old_topic = %d, ", 
                                token_entry->wordid,
                                accept,
                                new_topicid,
                                token_entry->topicid);
                        fprintf(stderr, "Ndt(new) = %d, Ntw(new) = %d, Nt(new) = %d, Pdw(new) = %.16f, Qw(new) = %.16f, ",
                                getDocTopicCnt(doc_topic_dist, num_topics, a, new_topicid),
                                getTopicWordCntAlias(topic_word_dist, num_topics, new_topicid, token_entry->wordid),
                                topic_row_sums[new_topicid],
                                dwbucket[new_topicid],
                                table->wbucket[new_topicid]);
                        fprintf(stderr, "Ndt(old) = %d, Ntw(old) = %d, Nt(old) = %d, Pdw(old) = %.16f, Qw(old) = %.16f\n",
                                getDocTopicCnt(doc_topic_dist, num_topics, a, token_entry->topicid),
                                getTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid),
                                topic_row_sums[token_entry->topicid],
                                dwbucket[token_entry->topicid],
                                table->wbucket[token_entry->topicid]);
                        fflush(stderr);
                        #endif
                    }
                }
            } else {
                // sample in common topic, topicid just num_topics
                token_entry->topicid = num_topics;
            }
            if (new_topicid < 0) {
                fprintf(stderr, "***ERROR***: sample fail, r = %.16f, P_dw = %.16f, Q_w = %.16f\n", r, P_dw, table->Q_w);
                fflush(stderr);
                exit(2);
            }
            
            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, 1);
            addTopicWordCntAlias(topic_word_dist, num_topics, token_entry->topicid, token_entry->wordid, 1);
            doc_entry->num_words++;
            topic_row_sums[token_entry->topicid]++;
            if (token_entry->topicid < num_topics) {
                // update sparse bucket
                updateDenomin(denominators, Vbeta, token_entry->topicid);
            }

        }
    }

    free(denominators);
    free(dwbucket);
    free(wbucket);
}

void saveModel(uint32 suffix) {
    uint32 a, b, cnt;
    int t;
    char fpath[MAX_STRING], word_str[MAX_STRING];
    FILE *fout;
    TopicNode *node;
    DocEntry *doc_entry;
    TokenEntry *token_entry;

    // save doc-topic
    sprintf(fpath, "%s/%s.%d", output, "doc_topic", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_docs; a++) {
        doc_entry = &doc_entries[a];
        node = doc_entry->nonzeros;
        while (node) {
            fprintf(fout, "%d:%d ", node->topicid, node->cnt);
            node = node->next;
        }
        fprintf(fout, "\n");
    }

    // save topic-word
    sprintf(fpath, "%s/%s.%d", output, "topic_word", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (t = 0; t < num_topics + 1; t++) {
        if (t == num_topics) fprintf(fout, "common-topic");
        else fprintf(fout, "topic-%d", t);
        for (b = 0; b < vocab_size; b++) {
            cnt = getTopicWordCntAlias(topic_word_dist, num_topics, t, b);
            if (cnt > 0) {
                getWordFromId(b, word_str);
                fprintf(fout, " %s:%d", word_str, cnt);
                memset(word_str, 0, MAX_STRING);
            }
        }
        fprintf(fout, "\n");
    }

    // save tokens
    sprintf(fpath, "%s/%s.%d", output, "tokens", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_docs; a++) {
        doc_entry = &doc_entries[a];
        for (b = 0; b < doc_entry->num_words; b++) {
            token_entry = &token_entries[doc_entry->idx + b];
            getWordFromId(token_entry->wordid, word_str);
            fprintf(fout, "%s:1:%d ", word_str, token_entry->topicid);
            memset(word_str, 0, MAX_STRING);
        }
        fprintf(fout, "\n");
    }
}

int main(int argc, char **argv) {
    int a;

    if (argc == 1) {
        printf("_____________________________________\n\n");
        printf("Hierarchy Latent Dirichlet Allocation\n\n");
        printf("_____________________________________\n\n");
        printf("Parameters:\n");
        printf("-input <file>\n");
        printf("\tpath of docs file, lines of file look like \"word1:freq1:topic1 word2:freq2:topic2 ... \\n\"\n");
        printf("\tword is <string>, freq is <int>, represent word-freqence in the document, topic is <int>, range from 0 to num_topics,\n");
        printf("\tused to anchor the word in the topicid, if you don't want to do that, set the topic < 0\n");
        printf("-output <dir>\n");
        printf("\tdir of model(word-topic, doc-topic) file\n");
        printf("-num_topics <int>\n");
        printf("\tnumber of topics\n");
        printf("-alpha <float>\n");
        printf("\tsymmetric doc-topic prior probability, default is 0.05\n");
        printf("-beta <float>\n");
        printf("\tsymmetric topic-word prior probability, default is 0.01\n");
        printf("-gamma0 <float>\n");
        printf("\t\"special topic\" prior probability, default is 0.1\n");
        printf("-beta2 <float>\n");
        printf("\t\"common topic\"-word prior probability, default is 0.01\n");
        printf("-num_iters <int>\n");
        printf("\tnumber of iteration, default is 20\n");
        printf("-save_step <int>\n");
        printf("\tsave model every save_step iteration, default is -1 (no save)\n");
        printf("-MH_step <int>\n");
        printf("\tnumber of MH sample, default is 2\n");
        return -1;
    }

    // parse args
    if ((a = argPos((char *)"-input", argc, argv)) > 0) {
        strcpy(input, argv[a + 1]);
    }
    if ((a = argPos((char *)"-output", argc, argv)) > 0) {
        strcpy(output, argv[a + 1]);
    }
    if ((a = argPos((char *)"-num_topics", argc, argv)) > 0) {
        num_topics = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-alpha", argc, argv)) > 0) {
        alpha = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-beta", argc, argv)) > 0) {
        beta = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-gamma", argc, argv)) > 0) {
        gamma0 = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-beta2", argc, argv)) > 0) {
        beta2 = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-num_iters", argc, argv)) > 0) {
        num_iters = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-save_step", argc, argv)) > 0) {
        save_step = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-MH_step", argc, argv)) > 0) {
        MH_step = atoi(argv[a + 1]);
    }

    topic_row_sums = (uint32 *)calloc(1 + num_topics, sizeof(uint32));
    memset(topic_row_sums, 0, (1 + num_topics) * sizeof(uint32));

    // load documents and allocate memory for entries
    srand(time(NULL));
    learnVocabFromDocs();
    loadDocs();

    // metropolis-hasting sampling
    printf("start train LDA:\n");
    fflush(stdout);
    initTable();
    printf("init table finished\n");
    for (a = 0; a < num_iters; a++) {
        if (save_step > 0 && a % save_step == 0) saveModel(a);
        MHSample(a);
    }

    // save model
    saveModel(num_iters);

    free(topic_row_sums);
    free(doc_topic_dist);
    free(topic_word_dist);
    free(doc_entries);
    free(alias_tables);
    free(token_entries);

    return 0;
}
