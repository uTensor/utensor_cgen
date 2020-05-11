# Instructions for generating Tensorflow Lite flatbuffer API

## Building flatc
Visit the Flatbuffer repository
Tutorials are here

```
git clone https://github.com/google/flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make
```

Add `flac` to your .bash_profile

```
alias flac=`~/PATH/TO/flatbuffers/flatc`
```


## Generate TF Lite schema

```
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
pip install flatbuffers
flatc -p schema.fbs
```