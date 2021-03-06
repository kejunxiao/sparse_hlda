#include "utils.h"
#include "model.h"
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <map>
#include <unordered_map>
#include <string>

/* gloabl variables */
// parameters
char input[MAX_STRING];
char output[MAX_STRING];
char init_doc_alpha[MAX_STRING];
char init_beta_word[MAX_STRING];
uint32 num_topics = 0;
real alpha = 0.05; // doc-topic prior
real beta = 0.01; // topic-word prior
real beta_common = 0.01;
real gamma0 = 0.1;
real eta = 0.1; // prior confidence coefficient
uint32 num_iters = 30;
int save_step = -1;

// train data related
uint32 num_docs = 0;
uint32 vocab_size = 0;
uint32 num_tokens = 0;
std::unordered_map<std::string, uint32> word2id;
std::unordered_map<uint32, std::string> id2word;
std::unordered_map<std::string, uint32> doc2id;
std::unordered_map<uint32, std::string> id2doc;

// model related
uint32 *topic_word_sums = NULL;
TopicNode *doc_topic_dist = NULL;
TopicNode *topic_word_dist = NULL;

DocEntry *doc_entries = NULL;
WordEntry *word_entries = NULL;
TokenEntry *token_entries = NULL;

std::map<std::pair<uint32, int>, uint32> init_doc_topic;
std::map<std::pair<int, uint32>, uint32> init_topic_word;

/* helper functions */
static void getWordFromId(uint32 wordid, char *word) {
    auto itr = id2word.find(wordid);
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
    auto itr = word2id.find(s);
    if (itr != word2id.end()) {
        return itr->second;
    } else { 
        word2id[s] = vocab_size;
        id2word[vocab_size] = s;
        vocab_size++;
        return word2id[s];
    }
}

static void getDocFromId(uint32 docid, char *doc) {
    auto itr = id2doc.find(docid);
    if (itr != id2doc.end()) {
        strcpy(doc, itr->second.c_str());
        return;
    } else {
        fprintf(stderr, "***ERROR***: unknown docid %d", docid);
        exit(1);
    }
}

static uint32 getIdFromDoc(const char *doc) {
    std::string s(doc);
    auto itr = doc2id.find(s);
    if (itr != doc2id.end()) {
        return itr->second;
    } else { 
        doc2id[s] = num_docs;
        id2doc[num_docs] = s;
        num_docs++;
        return doc2id[s];
    }
}

static int docIsExists(const char *doc) {
    std::string s(doc);
    auto itr = doc2id.find(s);
    if (itr != doc2id.end()) {
        return 1;
    } else {
        return 0;
    }
}

static int wordIsExists(const char *word) {
    std::string s(word);
    auto itr = word2id.find(s);
    if (itr != word2id.end()) {
        return 1;
    } else {
        return 0;
    }
}

inline static int genRandTopicId() { return rand() % num_topics; }

/* sparse LDA process */
// denominators
static void initDenomin(real *denominators, real Vbeta) {
    int t;
    for (t = 0; t < num_topics; t++) denominators[t] = Vbeta + topic_word_sums[t];
}

inline static void updateDenomin(real *denominators, real Vbeta, int topicid) {
    denominators[topicid] = Vbeta + topic_word_sums[topicid];
}

// soomth-only bucket
static real initS(real *sbucket, real ab, real *denominators) {
    int t;
    real smooth = 0;

    for (t = 0; t < num_topics; t++) {
        sbucket[t] = ab / denominators[t];
        smooth += sbucket[t];
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
inline static real initComm(real Vbeta_common, uint32 wordid) {
    return (getTopicWordCnt(topic_word_dist, num_topics, num_topics, wordid) + beta_common) / (topic_word_sums[num_topics] + Vbeta_common);
}

/* public interface */
void learnVocabFromDocs() {
    uint32 len, isdoc;
    char ch, *token, buf[MAX_STRING];
    FILE *fin;

    if (NULL == (fin = fopen(input, "r"))) {
        fprintf(stderr, "***ERROR***: can not open input file");
        exit(1);
    }
    // get number of documents and number of tokens from input file
    len = 0;
    isdoc = 1;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            buf[len] = '\0';
            if (isdoc == 1) {
                getIdFromDoc(buf);
                isdoc = 0;
            } else {
                token = strtok(buf, ":");  // get word-string
                getIdFromWord(token);
                token = strtok(NULL, ":"); // get word-freq
                num_tokens += atoi(token);
            }
            if (ch == '\n') {
                if (num_docs % 1000 == 0) {
                    printf("%dK%c", num_docs / 1000, 13);
                    fflush(stdout);
                }
                isdoc = 1;
            }
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
    printf("number of documents: %d, number of tokens: %d, vocabulary size: %d\n", num_docs, num_tokens, vocab_size);
}

void allocMem() {
    uint32 a;

    // allocate memory for topic_word_sums
    topic_word_sums = (uint32 *)calloc(1 + num_topics, sizeof(uint32));
    memset(topic_word_sums, 0, (1 + num_topics) * sizeof(uint32));
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
    uint32 a, b, c, freq, len, wordid, docid, isdoc;
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
    isdoc = 1;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            buf[len] = '\0';
            if (isdoc == 1) {
                docid = getIdFromDoc(buf);
                isdoc = 0;
            } else {
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
                    if (topicid >= 0) {
                        token_entry->topicid = topicid;
                    } else {
                        token_entry->topicid = genRandTopicId();
                    }
                    addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, 1);
                    addTopicWordCnt(topic_word_dist, num_topics, token_entry->topicid, &word_entries[wordid], 1);
                    topic_word_sums[token_entry->topicid]++;
                }
                c += freq;
            }
            if (ch == '\n') {
                if (docid % 1000 == 0) {
                    printf("%dK%c", docid / 1000, 13);
                    fflush(stdout);
                }
                doc_entry = &doc_entries[docid];
                doc_entry->idx = b;
                doc_entry->num_words = c - b;

                b = c;
                isdoc = 1;
            }
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
}

void loadInitDocPrior() {
    char ch, buf[MAX_STRING], *token;
    int topicid;
    uint32 len, isdoc, docid, num_read, cnt;
    FILE *fin;
    DocEntry *doc_entry;

    // load doc-alpha
    if (NULL == (fin = fopen(init_doc_alpha, "r"))) {
        fprintf(stderr, "***ERROR***: open %s fail", init_doc_alpha);
        exit(1);
    }
    printf("start load init-doc-alpha:\n");

    len = 0;
    isdoc = 1;
    num_read = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            buf[len] = '\0';
            if (isdoc == 1) {
                if (docIsExists(buf)) {
                    docid = getIdFromDoc(buf);
                } else {
                    while ('\n' != ch) {
                        ch = fgetc(fin);
                    }
                }
                isdoc = 0;
            } else {
                token = strtok(buf, ":"); // get topicid
                topicid = atoi(token);
                cnt = atoi(strtok(NULL, ":")); // get cnt
                cnt = eta * cnt;
                init_doc_topic[std::make_pair(docid, topicid)] = cnt;
                doc_entry = &doc_entries[docid];
                addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, topicid, cnt);
            }
            if (ch == '\n') {
                if (num_read % 1000 == 0) {
                    printf("%dK%c", num_read / 1000, 13);
                    fflush(stdout);
                }
                num_read++;
                isdoc = 1;
            }
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
    printf("init-doc-alpha load done.\n");
}

void loadInitWordPrior() {
    char ch, buf[MAX_STRING], *token;
    int topicid;
    uint32 len, istopic, wordid, num_read, cnt;
    FILE *fin;
    WordEntry *word_entry;

    // load beta-word
    if (NULL == (fin = fopen(init_beta_word, "r"))) {
        fprintf(stderr, "***ERROR***: open %s fail", init_beta_word);
        exit(1);
    }
    printf("start load init-beta-word:\n");

    topicid = -1;
    len = 0;
    istopic = 1;
    num_read = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            buf[len] = '\0';
            if (istopic == 1) {
                topicid = atoi(buf);
                if (topicid < 0 || topicid > num_topics) {
                    printf("topicid = %d, not in range [0, num_topics]\n", topicid);
                    exit(1);
                }
                istopic = 0;
            } else {
                if (num_read % 1000 == 0) {
                    printf("%dK%c", num_read / 1000, 13);
                    fflush(stdout);
                }
                token = strtok(buf, ":"); // get word
                if (wordIsExists(token)) {
                    wordid = getIdFromWord(token);
                    cnt = atoi(strtok(NULL, ":")); // get cnt
                    cnt = eta * cnt;
                    init_topic_word[std::make_pair(topicid, wordid)] = cnt;
                    word_entry = &word_entries[wordid];
                    addTopicWordCnt(topic_word_dist, num_topics, topicid, word_entry, cnt);
                    topic_word_sums[topicid] += cnt;
                    num_read++;
                }
            }
            if (ch == '\n') {
                istopic = 1;
            }
            memset(buf, 0, len);
            len = 0;
        } else { // append ch to buf
            if (len == MAX_STRING - 1) continue;
            buf[len] = ch;
            len++;
        }
    }
    printf("init-beta-word load done.\n");
}

void gibbsSample(uint32 round) {
    int new_topicid;
    uint32 a, b, n_spec;
    real smooth, dt, tw, spec_topic_r, s_spec, s_comm, r, s;
    real *denominators, *sbucket, *dbucket, *tbucket;
    real Kalpha = num_topics * alpha, Vbeta = vocab_size * beta, Vbeta_common = vocab_size * beta_common, ab = alpha * beta;
    struct timeval tv1, tv2;
    DocEntry *doc_entry;
    WordEntry *word_entry;
    TokenEntry *token_entry;
    TopicNode *node;

    // allocate memory for buckets
    denominators = (real *)calloc(num_topics, sizeof(real));
    sbucket = (real *)calloc(num_topics, sizeof(real));
    dbucket = (real *)calloc(num_topics, sizeof(real));
    tbucket = (real *)calloc(num_topics, sizeof(real));

    // init denominators
    memset(denominators, 0, num_topics * sizeof(real));
    initDenomin(denominators, Vbeta);
    // init soomth-only bucket
    memset(sbucket, 0, num_topics * sizeof(real));
    smooth = initS(sbucket, ab, denominators);

    gettimeofday(&tv1, NULL);
    // iterate docs
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
        // init doc-topic bucket
        memset(dbucket, 0, num_topics * sizeof(real));
        dt = initD(dbucket, doc_entry, denominators);
        // iterate tokens
        for (b = 0; b < doc_entry->num_words; b++) {
            token_entry = &token_entries[doc_entry->idx + b];
            word_entry = &word_entries[token_entry->wordid];

            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, token_entry->topicid, -1);
            addTopicWordCnt(topic_word_dist, num_topics, token_entry->topicid, word_entry, -1);
            doc_entry->num_words--;
            topic_word_sums[token_entry->topicid]--;

            if (token_entry->topicid < num_topics) {
                // only update special-topics
                updateDenomin(denominators, Vbeta, token_entry->topicid);
                smooth += updateS(sbucket, ab, denominators, token_entry->topicid);
                dt += updateD(dbucket, a, denominators, token_entry->topicid);
            }
            // init topic-word bucket
            memset(tbucket, 0, num_topics * sizeof(real));
            tw = initT(tbucket, word_entry, a, denominators);

            n_spec = doc_entry->num_words - getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics);
            spec_topic_r = (gamma0 + n_spec) / (1 + doc_entry->num_words);

            s_spec = spec_topic_r * (smooth + dt + tw) / (Kalpha + n_spec);
            s_comm = (1. - spec_topic_r) * initComm(Vbeta_common, token_entry->wordid);
            r = (s_spec + s_comm) * rand() / RAND_MAX;
            // start sampling
            new_topicid = -1;
            s = 0;
            if (r < s_spec) { 
                // sample in special topics, topicid range [0, num_topics - 1]
                r = (smooth + dt + tw) * rand() / (RAND_MAX + 1.);
                if (r < smooth) { // smooth-only bucket
                    for (new_topicid = 0; new_topicid < num_topics; new_topicid++) {
                        s += sbucket[new_topicid];
                        if (s > r) break;
                    }
                } else if (r < smooth + dt) { // doc-topic bucket
                    r -= smooth;
                    node = doc_entry->nonzeros;
                    while (node) {
                        s += dbucket[node->topicid];
                        if (s > r) {new_topicid = node->topicid; break;}
                        node = node->next;
                    }
                } else { // topic-word bucket
                    r -= smooth + dt;
                    node = word_entry->nonzeros;
                    while (node) {
                        s += tbucket[node->topicid];
                        if (s > r) {new_topicid = node->topicid; break;}
                        node = node->next;
                    }
                }
            } else { 
                // sample in common topic, topicid just num_topics
                new_topicid = num_topics;
            }
            if (new_topicid < 0) {
                fprintf(stderr, "***ERROR***: sample fail, r = %.16f, smooth = %.16f, dt = %.16f, tw = %.16f, s = %.16f\n", r, smooth, dt, tw, s);
                fflush(stderr);
                exit(1);
            }
            addDocTopicCnt(doc_topic_dist, num_topics, doc_entry, new_topicid, 1);
            addTopicWordCnt(topic_word_dist, num_topics, new_topicid, word_entry, 1);
            doc_entry->num_words++;
            topic_word_sums[new_topicid]++;
            if (new_topicid < num_topics) {
                // update sparse bucket
                updateDenomin(denominators, Vbeta, new_topicid);
                smooth += updateS(sbucket, ab, denominators, new_topicid);
                dt += updateD(dbucket, a, denominators, new_topicid);
            }

            token_entry->topicid = new_topicid;
        }
    }

    free(denominators);
    free(sbucket);
    free(dbucket);
    free(tbucket);
}

void saveModel(uint32 suffix) {
    uint32 a, b, cnt;
    int t;
    char fpath[MAX_STRING], doc_str[MAX_STRING], word_str[MAX_STRING];
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
        getDocFromId(a, doc_str);
        fprintf(fout, "%s", doc_str);
        doc_entry = &doc_entries[a];
        node = doc_entry->nonzeros;
        while (node) {
            cnt = node->cnt;
            auto itr = init_doc_topic.find(std::make_pair(a, node->topicid));
            if (itr != init_doc_topic.end()) {
                cnt -= itr->second;
            }
            if (cnt > 0) {
                fprintf(fout, " %d:%d", node->topicid, cnt);
            }
            node = node->next;
        }
        cnt = getDocTopicCnt(doc_topic_dist, num_topics, a, num_topics);
        auto itr = init_doc_topic.find(std::make_pair(a, num_topics));
        if (itr != init_doc_topic.end()) {
            cnt -= itr->second;
        }
        if (cnt > 0) {
            fprintf(fout, " %d:%d", num_topics, cnt);
        }
        fprintf(fout, "\n");
        memset(doc_str, 0, MAX_STRING);
    }
    fflush(fout);

    // save topic-word
    sprintf(fpath, "%s/%s.%d", output, "topic_word", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (t = 0; t < 1 + num_topics; t++) {
        fprintf(fout, "%d", t); // common-topic == 1 + num_topics
        for (b = 0; b < vocab_size; b++) {
            cnt = getTopicWordCnt(topic_word_dist, num_topics, t, b);
            auto itr = init_topic_word.find(std::make_pair(t, b));
            if (itr != init_topic_word.end()) {
                cnt -= itr->second;
            }
            if (cnt > 0) {
                getWordFromId(b, word_str);
                fprintf(fout, " %s:%d", word_str, cnt);
                memset(word_str, 0, MAX_STRING);
            }
        }
        fprintf(fout, "\n");
    }
    fflush(fout);

    // save tokens
    sprintf(fpath, "%s/%s.%d", output, "tokens", suffix);
    if (NULL == (fout = fopen(fpath, "w"))) {
        fprintf(stderr, "***ERROR***: open %s fail", fpath);
        exit(1);
    }
    for (a = 0; a < num_docs; a++) {
        getDocFromId(a, doc_str);
        fprintf(fout, "%s", doc_str);
        doc_entry = &doc_entries[a];
        for (b = 0; b < doc_entry->num_words; b++) {
            token_entry = &token_entries[doc_entry->idx + b];
            getWordFromId(token_entry->wordid, word_str);
            fprintf(fout, " %s:1:%d", word_str, token_entry->topicid);
            memset(word_str, 0, MAX_STRING);
        }
        fprintf(fout, "\n");
        memset(doc_str, 0, MAX_STRING);
    }
    fflush(fout);
}

void freeMem() {
    free(topic_word_sums);
    free(doc_topic_dist);
    free(topic_word_dist);
    free(doc_entries);
    free(word_entries);
    free(token_entries);
}

int main(int argc, char **argv) {
    int a;

    srand(time(NULL));

    if (argc == 1) {
        printf("_____________________________________\n\n");
        printf("Hierarchy Latent Dirichlet Allocation\n\n");
        printf("_____________________________________\n\n");
        printf("Parameters:\n");

        printf("-input <file>\n");
        printf("\tpath of docs file, lines of file look like \"doc word1:freq1:topic1 word2:freq2:topic2 ... \\n\"\n");
        printf("\tdoc is <string>, word is <string>, freq is <int>, represent word-freqence in the document, topic is <int>, range from [0, num_topics],\n");
        printf("\tused to anchor the word in the topicid, if you don't want to do that, set the topic < 0\n");

        printf("-init_doc_alpha <file>\n");
        printf("\tpath of init doc-alpha distribution file\n");

        printf("-init_beta_word <file>\n");
        printf("\tpath of init beta-word distribution file\n");

        printf("-output <dir>\n");
        printf("\tdir of model(doc-topic, doc-alpha, topic-word, beta-word) file\n");

        printf("-num_topics <int>\n");
        printf("\tnumber of special topics, special topicid range from [0, num_topics), common topicid is num_topics\n");

        printf("-alpha <float>\n");
        printf("\tsymmetric doc-topic prior probability, default is 0.05\n");

        printf("-beta <float>\n");
        printf("\tsymmetric topic-word prior probability, default is 0.01\n");

        printf("-beta_common <float>\n");
        printf("\t\"common topic\"-word prior probability, default is 0.01\n");

        printf("-gamma0 <float>\n");
        printf("\t\"special topic\" prior probability, default is 0.1\n");

        printf("-eta <float>\n");
        printf("\tlearning rate of asymmetric incremental doc-alpha/beta-word prior, default is 0.1\n");

        printf("-num_iters <int>\n");
        printf("\tnumber of iteration, default is 30\n");

        printf("-save_step <int>\n");
        printf("\tsave model every save_step iteration, default is -1 (no save)\n");

        return -1;
    }

    // parse args
    if ((a = argPos((char *)"-input", argc, argv)) > 0) {
        strcpy(input, argv[a + 1]);
    }
    if ((a = argPos((char *)"-init_doc_alpha", argc, argv)) > 0) {
        strcpy(init_doc_alpha, argv[a + 1]);
    }
    if ((a = argPos((char *)"-init_beta_word", argc, argv)) > 0) {
        strcpy(init_beta_word, argv[a + 1]);
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
    if ((a = argPos((char *)"-beta_common", argc, argv)) > 0) {
        beta_common = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-gamma0", argc, argv)) > 0) {
        gamma0 = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-eta", argc, argv)) > 0) {
        eta = atof(argv[a + 1]);
    }
    if ((a = argPos((char *)"-num_iters", argc, argv)) > 0) {
        num_iters = atoi(argv[a + 1]);
    }
    if ((a = argPos((char *)"-save_step", argc, argv)) > 0) {
        save_step = atoi(argv[a + 1]);
    }

    // load documents and allocate memory for entries
    learnVocabFromDocs();
    allocMem();
    loadDocs();

    // load init prior
    if (strlen(init_doc_alpha) > 0 && strlen(init_beta_word) > 0) {
        loadInitDocPrior();
        loadInitWordPrior();
    }

    // gibbs sampling
    printf("start train LDA:\n");
    for (a = 0; a < num_iters; a++) {
        if (save_step > 0 && a % save_step == 0) saveModel(a);
        gibbsSample(a);
    }

    // save model
    saveModel(num_iters);

    freeMem();

    return 0;
}
