# sparse_hlda
a fast Cpp-implementation Hierarchy Latent Dirichlet Allocation algorithm, can aggregate stop-words/meaningless-high-frequency-words into "common-topic"(a rubbish words bucket), generate K(number of topics you set) more pure "special-topics".

# features:
* supprot load last-trained-model and continue training;
* using sparse-gibbs-sampler, faster than collapsed-gibbs-sampler;
* using Hierarchy LDA structure, can aggregate stop-words/meaningless-high-frequency-words into "common-topic"(a rubbish words bucket);
* (developing)support mixture data structure (most freqence words saveing in continuous-memory and others saving in linked-list) to save memory;

# usage:  
./spare_hlda -input docs.txt -output model_out/ -num_topics 100 -num_iters 30 -save_step 10  
