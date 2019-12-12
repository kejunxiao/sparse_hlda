model_out=../data/fast_hlda_out
mkdir -p $model_out
../bin/fast_hlda -input ../data/train_input -output $model_out -num_topics 512 -save_step 10
