#include <iostream>
#include <vector>
#include <map>
#include <string>

struct element{
    int i;
    int j;
    double rate;
};


class PLSA_NMF{
protected:
    double** Pw_z;
    double** Pz_d;
    double*** Pz_wd;
    std::map< int,std::map<int, int> > n_dw;
    int row;
    int column;
    int K;
public:
    void init();
    void load_corpus(char*);
    void E_step();
    void M_step();
    double calcLogLiklihood();
    void var_infer(int step_count=5000, double alpha=0.0002, double beta=0.02, double threshold=0.01);
    double dot(int, int);//P[i][:] dot Q[:][j] 
    void model_output(char*);
    void train(char*);
    void predict(char*,char*);
    void debug_model_para();
};
