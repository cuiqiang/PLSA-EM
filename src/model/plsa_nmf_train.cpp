#include "plsa_nmf.h"

int main(int argc, char* argv[]){

    PLSA_NMF plsa_nmf;
    plsa_nmf.train(argv[1]);
}
