/* ========================================================
 *   Copyright (C) 2019 All rights reserved.
 *   
 *   filename : fast_hlda.c
 *   author   : ***
 *   date     : 2019-12-02
 *   info     : 
 * ======================================================== */
#include "utils.h"
#include "model.h"
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
uint32 num_words = 0;
std::unordered_map<std::string, uint32> word2id;
std::unordered_map<uint32, std::string> id2word;

// model related
DocEntry *doc_entries = NULL;
WordEntry *word_entries = NULL;
TopicEntry *topic_entries = NULL;

/* helper functions */
static void getWordFromId(uint32 wordid, char *word) {
    std::unordered_map<uint32, std::string>::iterator itr = id2word.find(wordid);
    if (itr != id2word.end()) {
        strcpy(word, itr->second.c_str());
        return;
    } else { 
        fprintf(stderr, "ERROR: unknown wordid %d", wordid); 
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

static int genRandTopicId() { return rand() % num_topics; }

/* sparse LDA process */
// soomth-only bucket
static real initS(real *sbucket) {
    int a;
    real smooth = 0, ab = alpha * beta, Vbeta = vocab_size * beta;

    for (a = 0; a < num_topics; a++) {
        sbucket[a] = ab / (Vbeta + topic_entries[a].num_words);
        smooth += sbucket[a];
    }
    return smooth;
}

static real updateS(real *sbucket, int oldtid, int newtid) {
    real delta = 0, tmp = 0, ab = alpha * beta, Vbeta = vocab_size * beta;

    // update old topicid
    tmp = ab / (Vbeta + topic_entries[oldtid].num_words);
    delta += tmp - sbucket[oldtid];
    sbucket[oldtid] = tmp;
    // update new topicid
    tmp = ab / (Vbeta + topic_entries[newtid].num_words);
    delta += tmp - sbucket[newtid];
    sbucket[newtid] = tmp;
    return delta;
}

// doc-topic bucket
static real initD(real *dbucket, DocEntry *doc_entry) {
    int a;
    real dt = 0, Vbeta = vocab_size * beta;

    for (a = 0; a < num_topics; a++) {
        if (getDocTopicCnt(doc_entry, a) == 0) continue;
        dbucket[a] = getDocTopicCnt(doc_entry, a) * beta / (Vbeta + topic_entries[a].num_words);
        dt += dbucket[a];
    }
    return dt;
}

static real updateD(real *dbucket, int oldtid, int newtid, DocEntry *doc_entry) {
    real delta = 0, tmp = 0, Vbeta = vocab_size * beta;

    // update old topicid
    tmp = getDocTopicCnt(doc_entry, oldtid) * beta / (Vbeta + topic_entries[oldtid].num_words);
    delta += tmp - dbucket[oldtid];
    dbucket[oldtid] = tmp;
    // update new topicid
    tmp = getDocTopicCnt(doc_entry, newtid) * beta / (Vbeta + topic_entries[newtid].num_words);
    delta += tmp - dbucket[newtid];
    dbucket[newtid] = tmp;
    return delta;
}

// topic-word bucket
static real initT(real *tbucket, DocEntry *doc_entry, WordEntry *word_entry) {
    int a;
    real tw = 0, Vbeta = vocab_size * beta;

    for (a = 0; a < num_topics; a++) {
        if (getTopicWordCnt(&topic_entries[a], word_entry->wordid) == 0) continue;
        tbucket[a] = (alpha + getDocTopicCnt(doc_entry, a)) * getTopicWordCnt(&topic_entries[a], word_entry->wordid) / (Vbeta + topic_entries[a].num_words);
        tw += tbucket[a];
    }
    return tw;
}

// common-word bucket
static real initComm(WordEntry *word_entry) {
    real Vbeta2 = vocab_size * beta2;
    return (getTopicWordCnt(&topic_entries[num_topics], word_entry->wordid) + beta2) / (topic_entries[num_topics].num_words + Vbeta2);
}

/* public interface */
void learnVocabFromDocs() {
    uint32 a, len;
    char ch, *token, buf[MAX_STRING];
    FILE *fin;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "can not open input file");
        exit(1);
    }
    // get number of documents and number of words from input file
    len = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == '\n') {
            num_docs++;
            if (num_docs % 1000 == 0) {
                printf("%dK%c", num_docs / 1000, 13);
                fflush(stdout);
            }
        } else if (ch == ' ') {
            buf[len] = '\0';
            token = strtok(buf, ":");  // get word-string
            getIdFromWord(token);
            token = strtok(NULL, ":"); // get word-freq
            num_words += atoi(token);
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            buf[len] = ch;
            len++;
        }
    }
    // allocate memory for doc_entries
    doc_entries = (DocEntry *)calloc(num_docs, sizeof(DocEntry));
    for (a = 0; a < num_docs; a++) docEntryInit(&doc_entries[a], a, num_topics);
    // allocate memory for word_entries
    word_entries = (WordEntry *)calloc(num_words, sizeof(WordEntry));
    // allocate memory for topic_entries
    topic_entries = (TopicEntry *)calloc(num_topics + 1, sizeof(TopicEntry));
    for (a = 0; a < num_topics + 1; a++) topicEntryInit(&topic_entries[a], a, vocab_size);

    printf("number of documents: %d, number of words: %d, vocabulary size: %d\n", num_docs, num_words, vocab_size);
}

void loadDocs() {
    int topicid;
    uint32 a, b, c, freq, len, wordid, docid;
    char ch, buf[MAX_STRING], *token;
    FILE *fin;
    DocEntry *doc_entry;
    WordEntry *word_entry;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "can not open input file");
        exit(1);
    }
    // load documents
    docid = 0;
    len = 0;
    b = 0;
    c = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
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
        } else if (ch == ' ') {
            buf[len] = '\0';
            token = strtok(buf, ":");  // get word-string
            wordid = getIdFromWord(token);  
            token = strtok(NULL, ":"); // get word-freq
            freq = atoi(token);
            token = strtok(NULL, ":"); // get anchor-topic
            topicid = atoi(token);

            doc_entry = &doc_entries[docid];
            for (a = 0; a < freq; a++) {
                word_entry = &word_entries[c + a];
                word_entry->wordid = wordid;
                if (topicid >= 0) word_entry->topicid = topicid;
                else word_entry->topicid = genRandTopicId();
                addTopicWordCnt(&topic_entries[word_entry->topicid], wordid, 1);
                addDocTopicCnt(doc_entry, word_entry->topicid, 1);
            }
            c += freq;
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            buf[len] = ch;
            len++;
        }
    }
}

void gibbsSample() {
    uint32 a, b;
    int c;
    real Kalpha, smooth, dt, tw, spec_topic_r, s_spec, s_comm, r, s, *sbucket, *dbucket, *tbucket;
    DocEntry *doc_entry;
    WordEntry *word_entry;

    Kalpha = num_topics * alpha;
    sbucket = (real *)calloc(num_topics, sizeof(real));
    dbucket = (real *)calloc(num_topics, sizeof(real));
    tbucket = (real *)calloc(num_topics, sizeof(real));

    smooth = initS(sbucket);
    for (a = 0; a < num_docs; a++) {
        if (a % 1000 == 0) {
            printf("%dK%c", a / 1000, 13);
            fflush(stdout);
        }
        doc_entry = &doc_entries[a];
        dt = initD(dbucket, doc_entry);
        for (b = 0; b < doc_entry->num_words; b++) {
            word_entry = &word_entries[doc_entry->idx + b];
            addTopicWordCnt(&topic_entries[word_entry->topicid], word_entry->wordid, -1);
            addDocTopicCnt(doc_entry, word_entry->topicid, -1);
            doc_entry->num_words--;

            spec_topic_r = (gamma0 + doc_entry->num_words - getDocTopicCnt(doc_entry, num_topics)) / (1 + doc_entry->num_words);

            tw = initT(tbucket, doc_entry, word_entry);
            s_spec = (smooth + dt + tw) * spec_topic_r / (Kalpha + doc_entry->num_words - getDocTopicCnt(doc_entry, num_topics));
            s_comm = (1. - spec_topic_r) * initComm(word_entry);
            r = (s_spec + s_comm) * rand() / RAND_MAX;
            if (r < s_spec) { // sample in special topics, topicid range 0 ~ num_topics - 1
                r = r / spec_topic_r * (Kalpha + doc_entry->num_words - getDocTopicCnt(doc_entry, num_topics));
                s = 0;
                if (r < smooth) {
                    for (c = 0; c < num_topics; c++) {
                        s += sbucket[c];
                        if (s > r) break;
                    }
                } else if (r < smooth + dt) {
                    for (c = 0; c < num_topics; c++) {
                        if (getDocTopicCnt(doc_entry, c) == 0) continue;
                        s += dbucket[c];
                        if (s > r) break;
                    }
                } else {
                    for (c = 0; c < num_topics; c++) {
                        if (getTopicWordCnt(&topic_entries[c], word_entry->wordid) == 0) continue;
                        s += tbucket[c];
                        if (s > r) break;
                    }
                }
            } else { // sample in common topic, topicid just num_topics
                c = num_topics;
            }
            if (c != word_entry->topicid) {
                // update topic-word
                addTopicWordCnt(&topic_entries[c], word_entry->wordid, 1);
                // update doc-topic
                addDocTopicCnt(doc_entry, c, 1);
                // update sparse bucket
                smooth += updateS(sbucket, word_entry->topicid, c);
                dt += updateD(dbucket, word_entry->topicid, c, doc_entry);
            }
            word_entry->topicid = c;
            doc_entry->num_words++;
        }
    }

    free(sbucket);
    free(dbucket);
    free(tbucket);
}

void saveModel(uint32 suffix) {
    uint32 a, b, cnt;
    int t;
    char fpath[MAX_STRING], word_str[MAX_STRING];
    FILE *fout;
    DocEntry *doc_entry;
    TopicEntry *topic_entry;
    WordEntry *word_entry;

    // save doc-topic
    sprintf(fpath, "%s/%s.%d", output, "doc_topic", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "ERRPR: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_docs; a++) {
        doc_entry = &doc_entries[a];
        for (t = 0; t < num_topics; t++) {
            cnt = getDocTopicCnt(doc_entry, t);
            if (cnt > 0) fprintf(fout, " %d:%d", t, cnt);
        }
        fprintf(fout, "\n");
    }

    // save topic-word
    sprintf(fpath, "%s/%s.%d", output, "topic_word", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "ERRPR: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_topics + 1; a++) {
        topic_entry = &topic_entries[a];
        if (a == num_topics) fprintf(fout, "common-topic");
        else fprintf(fout, "topic-%d", a);
        for (b = 0; b < vocab_size; b++) {
            cnt = getTopicWordCnt(topic_entry, b);
            if (cnt > 0) {
                getWordFromId(b, word_str);
                fprintf(fout, " %s:%d", word_str, cnt);
                memset(word_str, 0, MAX_STRING);
            }
        }
        fprintf(fout, "\n");
    }

    // save words
    sprintf(fpath, "%s/%s.%d", output, "words", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "ERRPR: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_docs; a++) {
        doc_entry = &doc_entries[a];
        for (b = 0; b < doc_entry->num_words; b++) {
            word_entry = &word_entries[doc_entry->idx + b];
            getWordFromId(word_entry->wordid, word_str);
            fprintf(fout, " %s:1:%d", word_str, word_entry->topicid);
            memset(word_str, 0, MAX_STRING);
        }
        fprintf(fout, "\n");
    }
}

int main(int argc, char **argv) {
    int a, sec1, sec2;

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


    // load documents and allocate memory for entries
    learnVocabFromDocs();
    loadDocs();

    // gibbs sampling
    printf("start train LDA:\n");
    for (a = 0; a < num_iters; a++) {
        if (save_step > 0 && a % save_step == 0) saveModel(a);
        sec1 = time(NULL);
        gibbsSample();
        sec2 = time(NULL);
        printf("iter %d done, take %d second\n", a, sec2 - sec1);
    }

    // save model
    saveModel(num_iters);

    for (a = 0; a < num_docs; a++) docEntryDestory(&doc_entries[a]);
    free(doc_entries);
    free(word_entries);
    for (a = 0; a < num_topics + 1; a++) topicEntryDestory(&topic_entries[a]);
    free(topic_entries);

    return 0;
}
