#!/bin/bash

INTER_DIR=intermediate_files
CORPUS=/fs/meili0/faheem/gpanlp/data/lemma_sent_uniq
VOCAB_FILE=$INTER_DIR/glove_vocab.txt
BUILDDIR=build
VERBOSE=2
MEMORY=100
VOCAB_MIN_COUNT=10
NUM_THREADS=8
X_MAX=100
MAX_ITER=30 # epochs
BINARY=2
window_sizes=( 8 )
vector_sizes=( 256 )
MODEL_DIR=../models_glove
echo

if [ ! -f $VOCAB_FILE ]; then

  echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
  $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

fi

for WINDOW_SIZE in ${window_sizes[@]}
  do

    echo "Window size is $WINDOW_SIZE"
    COOCCURRENCE_FILE=$INTER_DIR/cooccurrence.${WINDOW_SIZE}.bin
    if [ ! -f $COOCCURRENCE_FILE ]; then
      echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
      $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

    fi

    COOCCURRENCE_SHUF_FILE=$INTER_DIR/cooccurrence.shuf.${WINDOW_SIZE}.bin
    if [ ! -f $COOCCURRENCE_SHUF_FILE ]; then

      echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
      $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

    fi


  for VECTOR_SIZE in ${vector_sizes[@]}
    do
      echo "Vector dimension is $VECTOR_SIZE"
      SAVE_FILE=$MODEL_DIR/glove.w${WINDOW_SIZE}.d${VECTOR_SIZE}.model

      if [ ! -f $SAVE_FILE ]; then
        echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
        $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
      fi
  done
done




#
#
#
#
# if [ "$CORPUS" = 'text8' ]; then
#    if [ "$1" = 'matlab' ]; then
#        matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
#    elif [ "$1" = 'octave' ]; then
#        octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
#    else
#        echo "$ python eval/python/evaluate.py"
#        python eval/python/evaluate.py
#    fi
# fi
