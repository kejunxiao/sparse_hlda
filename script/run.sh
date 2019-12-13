model_out=../data/fast_hlda_out
mkdir -p $model_out
timeit ../bin/fast_hlda -input ../data/train_input -output $model_out -num_topics 256 -save_step 25 -num_iters 50
