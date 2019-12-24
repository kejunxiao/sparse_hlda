#include "utils.h"
#include "model.h"
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

// train data related
uint32 num_docs = 0;
uint32 vocab_size = 0;
uint32 num_tokens = 0;
std::unordered_map<std::string, uint32> word2id;
std::unordered_map<uint32, std::string> id2word;

// model related
uint32 *topic_row_sums = NULL;
TopicNode *doc_topic_dist = NULL;
TopicNode *topic_word_dist = NULL;

DocEntry *doc_entries = NULL;
WordEntry *word_entries = NULL;
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

/* sparse LDA process */
// denominators
static void initDenomin(real *denominators, real Vbeta) {
    int a;
    for (a = 0; a < num_topics; a++) denominators[a] = Vbeta + topic_row_sums[a];
}

inline static void updateDenomin(real *denominators, real Vbeta, int topicid) {
    denominators[topicid] = Vbeta + topic_row_sums[topicid];
}

// soomth-only bucket
static real initS(real *sbucket, real ab, real *denominators) {
    int a;
    real smooth = 0;

    for (a = 0; a < num_topics; a++) {
        sbucket[a] = ab / denominators[a];
        smooth += sbucket[a];
    }
    return smooth;
}

static real updateS(real *sbucket, real ab, real *denominators, int topicid) {
    real delta = 0, tmp = 0;

    tmp = ab / denominators[topicid];
    delta += tmp - sbucket[topicid];
    sbucket[topicid] = tmp;
    return delta;
}

// doc-topic bucket
static real initD(real *dbucket, DocEntry *doc_entry, real *denominators) {
    TopicNode *node;
    real dt = 0;

    node = doc_entry->nonzeros;
    while (node) {
        dbucket[node->topicid] = node->cnt * beta / denominators[node->topicid];
        dt += dbucket[node->topicid];
        node = node->next;
    }
    return dt;
}

static real updateD(real *dbucket, uint32 docid, real *denominators, int topicid) {
    real delta = 0, tmp = 0;

    tmp = getDocTopicCnt(doc_topic_dist, num_topics, docid, topicid) * beta / denominators[topicid];
    delta += tmp - dbucket[topicid];
    dbucket[topicid] = tmp;
    return delta;
}

// topic-word bucket
static real initT(real *tbucket, WordEntry *word_entry, uint32 docid, real *denominators) {
    TopicNode *node;
    real tw = 0;

    node = word_entry->nonzeros;
    while (node) {
        tbucket[node->topicid] = (alpha + getDocTopicCnt(doc_topic_dist, num_topics, docid, node->topicid)) * node->cnt / denominators[node->topicid];
        tw += tbucket[node->topicid];
        node = node->next;
    }
    return tw;
}

// common-word bucket
inline static real initComm(real Vbeta2, uint32 wordid) {
    return (getTopicWordCnt(topic_word_dist, num_topics, num_topics, wordid) + beta2) / (topic_row_sums[num_topics] + Vbeta2);
}

/* AliasLDA */

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
    topic_word_dist = (TopicNode *)calloc(vocab_size * (1 + num_topics), sizeof(TopicNode));
    for (a = 0; a < vocab_size * (1 + num_topics); a++) topicNodeInit(&topic_word_dist[a], a % (1 + num_topics));
    // allocate memory for doc_entries
    doc_entries = (DocEntry *)calloc(num_docs, sizeof(DocEntry));
    for (a = 0; a < num_docs; a++) docEntryInit(&doc_entries[a], a);
    // allocate memory for word_entries
    word_entries = (WordEntry *)calloc(vocab_size, sizeof(WordEntry));
    for (a = 0; a < vocab_size; a++) wordEntryInit(&word_entries[a], a);
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
                addTopicWordCnt(topic_word_dist, num_topics, token_entry->topicid, &word_entries[wordid], 1);
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

void MHSample(uint32 round) {
    uint32 a, b;
    int t;
    struct timeval tv1, tv2;
    real smooth, dt, tw, spec_topic_r, s_spec, s_comm, r, s, *denominators, *sbucket, *dbucket, *tbucket;
    real Kalpha = num_topics * alpha, Vbeta = vocab_size * beta, Vbeta2 = vocab_size * beta2, ab = alpha * beta;
    DocEntry *doc_entry;
    TokenEntry *token_entry;
    WordEntry *word_entry;
    TopicNode *node;

    denominators = (real *)calloc(num_topics, sizeof(real));
    sbucket = (real *)calloc(num_topics, sizeof(real));
    dbucket = (real *)calloc(num_topics, sizeof(real));
    tbucket = (real *)calloc(num_topics, sizeof(real));

    memset(denominators, 0, num_topics);
    initDenomin(denominators, Vbeta);
    memset(sbucket, 0, num_topics);
    smooth = initS(sbucket, ab, denominators);
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
        memset(dbucket, 0, num_topics);
        dt = initD(dbucket, doc_entry, denominators);

        for (b = 0; b < doc_entry->num_words; b++) {
            token_entry = &token_entries[doc_entry->idx + b];
            word_entry = &word_entries[token_entry->wordid];

            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, -1);
            addTopicWordCnt(topic_word_dist, num_topics, token_entry->topicid, word_entry, -1);
            doc_entry->num_words--;
            topic_row_sums[token_entry->topicid]--;

            if (token_entry->topicid < num_topics) {
                // only update special-topics
                updateDenomin(denominators, Vbeta, token_entry->topicid);
                smooth += updateS(sbucket, ab, denominators, token_entry->topicid);
                dt += updateD(dbucket, a, denominators, token_entry->topicid);
            }
            memset(tbucket, 0, num_topics);
            tw = initT(tbucket, word_entry, a, denominators);

            spec_topic_r = (gamma0 + doc_entry->num_words - getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics)) / (1 + doc_entry->num_words);

            s_spec = (smooth + dt + tw) * spec_topic_r / (Kalpha + doc_entry->num_words - getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics));
            s_comm = (1. - spec_topic_r) * initComm(Vbeta2, token_entry->wordid);
            r = (s_spec + s_comm) * rand() / RAND_MAX;
            // start sampling
            t = -1;
            s = 0;
            if (r < s_spec) { 
                // sample in special topics, topicid range 0 ~ num_topics - 1
                r = (smooth + dt + tw) * rand() / (RAND_MAX + 1.);
                //printf("docid = %d, r = %.16f, smooth = %.16f, dt = %.16f, tw = %.16f\n", a, r, smooth, dt, tw);
                if (r < smooth) {
                    for (t = 0; t < num_topics; t++) {
                        s += sbucket[t];
                        if (s > r) break;
                    }
                } else if (r < smooth + dt) {
                    r -= smooth;
                    node = doc_entry->nonzeros;
                    while (node) {
                        s += dbucket[node->topicid];
                        if (s > r) {t = node->topicid; break;}
                        node = node->next;
                    }
                } else {
                    r -= smooth + dt;
                    node = word_entry->nonzeros;
                    while (node) {
                        s += tbucket[node->topicid];
                        if (s > r) {t = node->topicid; break;}
                        node = node->next;
                    }
                }
            } else { 
                // sample in common topic, topicid just num_topics
                //printf("docid = %d, r = %.16f, s_spec = %.16f, s_comm = %.16f\n", a, r, s_spec, s_comm);
                t = num_topics;
            }
            if (t < 0) {
                fprintf(stderr, "***ERROR***: sample fail, r = %.16f, smooth = %.16f, dt = %.16f, tw = %.16f\n", r, smooth, dt, tw);
                fprintf(stderr, "***ERROR***: node is NULL? %d, s = %.16f\n", node == NULL ? 1 : 0, s);
                for (int x = 0; x < num_topics + 1; x++) {
                    if (getDocTopicCnt(doc_topic_dist, num_topics, a, x) > 0) {
                        fprintf(stderr, "%d:%d ", x, getDocTopicCnt(doc_topic_dist, num_topics, a, x));
                    }
                }
                fprintf(stderr, "\n");
                float s2 = 0;
                for (int x = 0; x < num_topics; x++) {
                    if (dbucket[x] > 0) {
                        fprintf(stderr, "%d:%.16f ", x, dbucket[x]);
                        s2 += dbucket[x];
                    }
                }
                fprintf(stderr, "\ns2 = %.16f\n", s2);

                fflush(stderr);
                exit(2);
            }
            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, t, 1);
            addTopicWordCnt(topic_word_dist, num_topics, t, word_entry, 1);
            doc_entry->num_words++;
            topic_row_sums[t]++;
            if (t < num_topics) {
                // update sparse bucket
                updateDenomin(denominators, Vbeta, t);
                smooth += updateS(sbucket, ab, denominators, t);
                dt += updateD(dbucket, a, denominators, t);
            }

            token_entry->topicid = t;
        }
    }

    free(denominators);
    free(sbucket);
    free(dbucket);
    free(tbucket);
}

void saveModel(uint32 suffix) {
    uint32 a, b, cnt;
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
    for (a = 0; a < num_topics + 1; a++) {
        if (a == num_topics) fprintf(fout, "common-topic");
        else fprintf(fout, "topic-%d", a);
        for (b = 0; b < vocab_size; b++) {
            cnt = getTopicWordCnt(topic_word_dist, num_topics, a, b);
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
            fprintf(fout, " %s:1:%d", word_str, token_entry->topicid);
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

    topic_row_sums = (uint32 *)calloc(1 + num_topics, sizeof(uint32));
    memset(topic_row_sums, 0, (1 + num_topics) * sizeof(uint32));

    // load documents and allocate memory for entries
    srand(time(NULL));
    learnVocabFromDocs();
    loadDocs();

    // gibbs sampling
    printf("start train LDA:\n");
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
    free(word_entries);
    free(token_entries);

    return 0;
}
