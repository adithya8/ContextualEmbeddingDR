#author: @adithya8

declare -a msg=(14 36 64 118 207 386)
declare -a title=(6 14 26 47 83 154)
declare -a msgK=(14 36 71 143 286 357 495)
declare -a titleK=(6 14 29 57 114 143 198)
declare -a totK=(16 32 64 128 256 512 1024 2048)
declare -a totK_=( )


# Last array index shouldn't apply for XLNet

if [ "$#" -eq "3" ];
then
    experiment=$1
    contextualEmbedding=$2
    dimRedModel=$3
elif [ "$#" -eq "2" ]; 
then 
    experiment="19"
    contextualEmbedding=$1
    dimRedModel=$2
else
    echo "Pass Contextual Embedding (BERTB/XLNet), DimRedModel name as arg (pca/fa/nmf/nmfrand) -- Exiting !!!!"
    exit
fi

if [ "$1" -ne "19" ]; 
then
    msgK=( "${totK[@]}" )
    titleK=( "${totK_[@]}" )
fi

arraylength=${#msgK[@]}
for (( i=1; i<${arraylength}+1; i++ ));
do
    echo "bash ~/NLP/ContextualEmbeddingDR/${contextualEmbedding}DimRedExp.sh ${experiment} ${dimRedModel} ${msgK[$i-1]} ${titleK[$i-1]}"
    echo "----------------------------------------------------------------------"
    eval "bash ~/NLP/ContextualEmbeddingDR/${contextualEmbedding}DimRedExp.sh ${experiment} ${dimRedModel} ${msgK[$i-1]} ${titleK[$i-1]}"
done

#echo "f1 scores for increasing k sizes"
#eval  "cat ./results/XLNet_${dimRedModel}/* | grep \'f1\':"
#eval "python ~/NLP/ContextualEmbeddingDR/tableMaker.py ${experiment} ${contextualEmbedding} ${dimRedModel}"