#include "plsa_nmf.h"
#include "../common/constants.h"
#include "../common/util.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <sstream>

using namespace std;

void PLSA_NMF::init(){
	
	//init p(w|z),p(z|d)
	Pw_z = new double*[K];
    for(int z=0; z<K; z++){
    	double norm = 0.0;
        Pw_z[z] = new double[row];
        for(int w=0; w<row; w++){
        	double val = (double)rand();
        	Pw_z[z][w] = val;
        	norm += val;
        }

        for(int w=0; w<row; w++){
        	Pw_z[z][w]/=norm;
        	//cout << Pw_z[z][w] << SPACE;
        }
        //cout << endl;
    }
    
    Pz_d = new double*[column];
    for(int d=0; d<column; d++){
        Pz_d[d] = new double[K];
		double norm = 0.0;
		for(int z=0; z<K; z++){
			double val = (double)rand();
			Pz_d[d][z] = val;
			norm += val;
		}

		for(int z=0; z<K; z++){
			Pz_d[d][z]/=norm;
			//cout << Pd_z[z][d] << SPACE;
		}
		//cout << endl;
    }

    //init p(z|w,d)
    Pz_wd = new double**[row];
    for(int w=0; w<row; w++){
    	Pz_wd[w]=new double*[column];
    	for(int d=0; d<column; d++){
    		Pz_wd[w][d] = new double[K];
    	}
    }

   cout << "init p(w|z),p(z|d)" << endl;

}

void PLSA_NMF::E_step()// assign p(z|w,d)
{
	for(int w=0; w<row; w++){
		for(int d=0; d<column; d++){
			double norm = 0.0;
			for(int z=0; z<K; z++){
				double val = Pw_z[z][w]*Pz_d[d][z];
				Pz_wd[w][d][z] = val;
				norm += val;
			}
			for(int z=0; z<K; z++){
				Pz_wd[w][d][z]/=norm;
				if(Pz_wd[w][d][z] < MIN_VALUE){
					Pz_wd[w][d][z] = MIN_VALUE;
				}
				//cout << Pz_wd[w][d][z] << SPACE;
			}
			//cout << endl;
		}
		//cout << endl;
	}
}

void PLSA_NMF::M_step()
{
	for(int z=0; z<K; z++){
		
		//update p(w|z)
		double norm = 0.0;
		for(int w=0; w<row; w++){
			double sum = 0.0;
			for(int d=0; d<column; d++){		
				if(n_dw.find(d)!=n_dw.end() && n_dw[d].find(w)!=n_dw[d].end()){
					double val = n_dw[d][w]*Pz_wd[w][d][z];
					sum += val;
				}
			}
			Pw_z[z][w] = sum;
			norm += sum;
		}
		for(int w=0; w<row; w++){
			Pw_z[z][w] /= norm;
		}
	}

	//update p(z|d)
	for(int d=0; d<column; d++){
		double norm = 0.0;		
		for(int z=0; z<K; z++){
			double sum = 0.0;
			for(int w=0; w<row; w++){
				if(n_dw.find(d)!=n_dw.end() && n_dw[d].find(w)!=n_dw[d].end()){
					double val = n_dw[d][w]*Pz_wd[w][d][z];
					sum += val;
				}
			}
			Pz_d[d][z] = sum;
			norm += sum;
		}
		for(int z=0; z<K; z++){
			Pz_d[d][z] /= norm;
		}
	}

}

double PLSA_NMF::calcLogLiklihood()
{
	double loglik = 0.0;
	for(int w=0; w<row; w++){
		for(int d=0; d<column; d++){
			double sum = 0.0;
			for(int z=0; z<K; z++){
				sum += Pw_z[z][w]*Pz_d[d][z];
			}
			if(n_dw.find(d)!=n_dw.end() && n_dw[d].find(w)!=n_dw[d].end()){
				loglik += n_dw[d][w]*log(sum);
			}
		}
	}
	return loglik;

}
void PLSA_NMF::var_infer(int step_count, double alpha, double beta, double threshold)
{
    init();
    double L = 0.0;
    for(int step=0; step<step_count; step++){
    
        cout << "iteration: " << step << "\t";

        E_step();
        M_step();
		L = calcLogLiklihood();

		cout << L << endl;
   }
}

void PLSA_NMF::debug_model_para(){
	for(int w=0; w<row; w++){
		for(int z=0; z<K; z++){
			cout << Pw_z[z][w] << SPACE;
		}
		cout << endl;
	}
	for(int d=0; d<column; d++){
		for(int z=0; z<K; z++){
			cout << Pz_d[d][z] << SPACE;
		}
		cout << endl;
	}
}

void PLSA_NMF::model_output(char* modelFile){
    
    ofstream modelOut;
    modelOut.open(modelFile);

    modelOut << row << SPACE << K << SPACE << endl;
    for(int w=0; w<row; w++){
        for(int z=0; z<K; z++){
            modelOut << Pw_z[z][w] << SPACE;
        }
        modelOut << endl;
    }
    modelOut << K << SPACE << column << SPACE << endl;
    for(int d=0; d<column; d++){
        for(int z=0; z<K; z++){
            modelOut << Pz_d[d][z] << SPACE;
        }
        modelOut << endl;
    }
    modelOut << endl;

}

void PLSA_NMF::train(char* trainData){
    load_corpus(trainData);
    var_infer();
    char modelFile[200];
	strcpy(modelFile,trainData);
    model_output(strcat(modelFile,".model"));
}

void PLSA_NMF::load_corpus(char* trainData){
   
    ifstream in;
    in.open(trainData);

    in >> row >> column >> K;

    while(!in.eof()){
        string line;
        getline(in, line);
        if (line.size() == 0)
            continue;
        
        vector<string> secs = Util::split(line, SPACE);
        int d = atoi(secs[0].c_str())-1;
		int w = atoi(secs[1].c_str())-1;
        int n = atoi(secs[2].c_str());
        n_dw[d][w]=n;
    }
    cout << "load finish!" << endl;
}

