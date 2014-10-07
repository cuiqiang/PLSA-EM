CPP = g++
OFLAG = -o
COMPILE.cxx= $(CPP) -c

MODEL_PATH=./model
COMMON_PATH=./common
SOURCES= $(wildcard $(COMMON_PATH)/*.cpp) $(wildcard $(MODEL_PATH)/*.cpp)
OBJS_NAME=$(notdir $(patsubst %.cpp,%.o,$(SOURCES)))
OBJS=$(addprefix $(DEBUG_PATH)/,$(notdir $(patsubst %.cpp,%.o,$(SOURCES))))
BIN_PATH=../bin
DEBUG_PATH=../debug

PLSA_SVD_MODEL_TRAIN = util.o plsa_svd.o plsa_svd_train.o
PLSA_NMF_MODEL_TRAIN = util.o plsa_nmf.o plsa_nmf_train.o

PLSA_SVD=plsa_svd_train
PLSA_NMF=plsa_nmf_train

plsa_svd:$(addprefix $(BIN_PATH)/, $(PLSA_SVD))
plsa_nmf:$(addprefix $(BIN_PATH)/, $(PLSA_NMF))

# Rules for generating object files

$(OBJS):$(SOURCE)
	$(CPP) $(OFLAG) $@ -c $(filter %$(*F).cpp,$(SOURCES))

#
$(BIN_PATH)/plsa_svd_train: $(addprefix $(DEBUG_PATH)/, $(PLSA_SVD_MODEL_TRAIN))
	$(CPP) $(OFLAG) $@ $^
$(BIN_PATH)/plsa_nmf_train: $(addprefix $(DEBUG_PATH)/, $(PLSA_NMF_MODEL_TRAIN))
	$(CPP) $(OFLAG) $@ $^

clean:
	rm $(DEBUG_PATH)/*.o
