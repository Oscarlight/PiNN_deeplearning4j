cd deeplearning4j
# bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0'
bash buildmultiplescalaversions.sh install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0'
