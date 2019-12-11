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

// train data related
uint32 num_docs = 0;
uint32 vocab_size = 0;
uint64 num_words = 0;
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
        vocab_size++;
        word2id[s] = vocab_size;
        id2word[vocab_size] = s;
        return vocab_size;
    }
}

static int genRandTopicId() { return rand() % num_topics; }

/* sparse LDA process */
// soomth-only bucket
static real initS(real *sbucket) {}
// doc-topic bucket
static real initD(real *dbucket, DocEntry *doc_entry) {}
// topic-word bucket
static real initT(real *tbucket, DocEntry *doc_entry, WordEntry *word_entry) {}

/* public interface */
void learnVocabFromDocs() {
    uint32 len;
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
            token = strtok(buf, ":");  // get word-string
            token = strtok(NULL, ":"); // get word-freq
            num_words += atoi(token);
            memset(buf, 0, len);
        } else { // append ch to buf
            len = strlen(buf);
            buf[len] = ch;
            buf[len + 1] = '\0';
        }
    }
    // allocate memory for doc_entries and word_entries
    doc_entries = (DocEntry *)calloc(num_docs, sizeof(DocEntry));
    word_entries = (WordEntry *)calloc(num_words, sizeof(WordEntry));

    printf("number of documents: %d, number of words: %lld\n", num_docs, num_words);
}

void loadDocs() {
    int topicid;
    uint32 a, b, c, freq, len, wordid, docid;
    char ch, buf[MAX_STRING], *token;
    FILE *fin;
    DocEntry doc_entry;
    WordEntry word_entry;
    TopicEntry topic_entry;

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
            doc_entry = doc_entries[docid];
            doc_entry.docid = docid;
            doc_entry.idx = b;
            doc_entry.num_words = c - b;
            doc_entry.num_common_words = 0;

            docid++;
            b = c;
            if (docid % 1000 == 0) {
                printf("%dK%c", docid / 1000, 13);
                fflush(stdout);
            }
        } else if (ch == ' ') {
            token = strtok(buf, ":");  // get word-string
            wordid = getIdFromWord(token);  
            token = strtok(NULL, ":"); // get word-freq
            freq = atoi(token);
            c += freq;
            token = strtok(NULL, ":"); // get anchor-topic
            topicid = atoi(token);

            for (a = 0; a < freq; a++) {
                word_entry = word_entries[b + a];
                word_entry.wordid = wordid;
                if (topicid >= 0) word_entry.topicid = topicid;
                else word_entry.topicid = genRandTopicId();
                topic_entry = topic_entries[word_entry.topicid];
                topic_entry.num_words++;
                setTopicWordCnt(&topic_entry, wordid, getTopicWordCnt(&topic_entry, wordid) + 1);
            }
            memset(buf, 0, len);
        } else { // append ch to buf
            len = strlen(buf);
            buf[len] = ch;
            buf[len + 1] = '\0';
        }
    }
    printf("vocabulary size: %d\n", vocab_size);
}

void gibbsSample() {
    uint32 a, b, idx;
    real smooth, dt, tw, spec_topic_r, s_spec, s_comm, r, sbucket[num_topics], dbucket[num_topics], tbucket[num_topics];
    real Kalpha = num_topics * alpha;
    DocEntry doc_entry;
    WordEntry word_entry;

    smooth = initS(sbucket);
    for (a = 0; a < num_docs; a++) {
        doc_entry = doc_entries[a];
        dt = initD(dbucket, &doc_entry);
        for (b = 0; b < doc_entry.num_words; b++) {
            word_entry = word_entries[doc_entry.idx + b];
            tw = initT(tbucket, &doc_entry, &word_entry);
            spec_topic_r = (gamma0 + doc_entry.num_words - doc_entry.num_common_words) / (1 + doc_entry.num_words);
            s_spec = smooth + dt + tw;
            s_spec = spec_topic_r * s_spec / (Kalpha + doc_entry.num_words - doc_entry.num_common_words);
            s_comm = (1. - spec_topic_r) * initComm(&word_entry);
            r = (s_spec + s_comm) * rand() / RAND_MAX;
            if (s_spec > r) { // sample in special topics, topicid range 0 ~ num_topics - 1

            } else { // sample in common topic, topicid just num_topics

            }
        }
    }
}

void saveModel() {}

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

    // allocate memory for topic_entries
    topic_entries = (TopicEntry *)calloc(num_topics + 1, sizeof(TopicEntry));
    for (a = 0; a < num_topics + 1; a++) topicEntryInit(&topic_entries[a], a);

    // load documents and allocate memory for doc_entries and word_entries
    learnVocabFromDocs();
    loadDocs();

    // gibbs sampling
    for (a = 0; a < num_iters; a++) {
        sec1 = time(NULL);
        gibbsSample();
        sec2 = time(NULL);
        printf("iter %d done, take %d second\n", a, sec2 - sec1);
    }

    // save model
    saveModel();

    free(doc_entries);
    free(word_entries);
    for (a = 0; a < num_topics + 1; a++) topicEntryDestory(&topic_entries[a]);
    free(topic_entries);

    return 0;
}
