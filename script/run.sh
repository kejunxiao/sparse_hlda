model_out=../data/fast_hlda_out
mkdir -p $model_out
mkdir -p ../logs/
cd ../src && make clean && make -j && make install
time ../bin/fast_hlda -input ../data/train_input.tmp -output $model_out -num_topics 256 -save_step 25 -num_iters 50 > ../logs/log 2>&1 &
tail -f ../logs/log
