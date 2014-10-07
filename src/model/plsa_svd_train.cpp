#include "plsa_svd.h"

int main(int argc, char* argv[]){

    PLSA_SVD plsa_svd;
    plsa_svd.train(argv[1]);
}
