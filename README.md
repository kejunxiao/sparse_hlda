# fast_hlda
a fast Cpp-implementation Hierarchy Latent Dirichlet Allocation algorithm, can aggregate stop-words/meaningless-high-frequency-words into "common-topic"(a rubbish words bucket), generate K(number of topics you set) more pure "special-topics".

# features:
* supprot load last-trained-model and continue training;
* using sparse-gibbs-sampler, faster than vanilla-gibbs-sampler;
* using Hierarchy LDA structure, can aggregate stop-words/meaningless-high-frequency-words into "common-topic"(a rubbish words bucket);
* will support mixture data structure (most freqence words saveing in continuous-memory and others saving in linked-list) to save memory;

# arguments:
_____________________________________

Hierarchy Latent Dirichlet Allocation

_____________________________________

Parameters:
* -input <file>  
	path of docs file, lines of file look like "word1:freq1:topic1 word2:freq2:topic2 ... \n"  
	word is <string>, freq is <int>, represent word-freqence in the document, topic is <int>, range from 0 to num_topics,  
	used to anchor the word in the topicid, if you don't want to do that, set the topic < 0  
* -output <dir>  
	dir of model(word-topic, doc-topic) file  
* -num_topics <int>  
	number of topics  
* -alpha <float>  
	symmetric doc-topic prior probability, default is 0.05  
* -beta <float>  
	symmetric topic-word prior probability, default is 0.01  
* -gamma0 <float>  
	"special topic" prior probability, default is 0.1  
* -eta <float>  
	"common topic"-word prior probability, default is 0.01  
* -num_iters <int>  
	number of iteration, default is 20  
* -save_step <int>  
	save model every save_step iteration, default is -1 (no save)  

# usage:
./spare_hlda -input docs.txt -output model_out/ -num_topics 100 -num_iters 30 -save_step 10
