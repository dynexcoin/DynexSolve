// Copyright (c) 2021-2022, The Dynex Project
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other
//    materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors may be
//    used to endorse or promote products derived from this software without specific
//    prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <future>
#include <numeric>
#include <sstream>
#include <iostream>
#include <thread>
#include <cassert>
#include <atomic>
#include <random>
#include <chrono>

#include <iomanip>
#include <ctime>
#include <sstream>

// for ftp:
#include <curl/curl.h> //required for MPI - dependency
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// linux:
#ifndef WIN32
#include <climits>
#include <cstring>
#include <algorithm>
#endif

#include "log.hpp"

std::vector<std::string> splitString(const std::string& str, char delim);

// ODE integration types:
typedef double value_type;
typedef std::vector< value_type > state_type;

// debugging output:
//bool dynex_debugger false;

// protocol definitions:
#define PROT_JOB_POSTED 0x0a
#define PROT_JOB_CANCELLED 0x1a
#define PROT_JOB_ACCEPTED 0x0b
#define PROT_JOB_CREDENTIALS 0x1b
#define PROT_JOB_CONFIGURED 0x0c
#define PROT_JOB_RUNNING 0x0d
#define PROT_JOB_UPDATED 0x0e
#define PROT_JOB_FINISHED 0x0f

// state machine definitions:
#define DYNEX_STATE_OFF -1
#define DYNEX_STATE_IDLE 0
#define DYNEX_STATE_ACCEPTED 1
#define DYNEX_STATE_CREDENTIALS 2
#define DYNEX_STATE_CONFIGURED 3
#define DYNEX_STATE_RUNNING 4
#define DYNEX_STATE_FINISHED 5

// dynex chip definitions:
#define         MAX_LITS_SYSTEM 25
#define         PARAM_COUNT     9

// FTP server for jobs:
#define FTP_ADDRESS "ftp.dynexcoin.org"
#define FTP_PORT "21"


//---------------------------------------------------------------------------------------------------------------------------
// oberver & protocol handler
//---------------------------------------------------------------------------------------------------------------------------

class dynex_chip_thread_obj {
	
	std::promise<void> exitSignal;
    std::future<void> futureObj;

	public:
		//
		bool dynex_debugger = false;
		// job vars:
		int job_user_id;
    	int job_id;
    	uint64_t job_maxrate;
    	int job_type;
    	int job_dynexchips;
    	int job_available_slot;
    	uint64_t job_max_walltime;
    	std::string job_input_file;
    	// input file vars:
    	int n;
    	int m;
    	int * cls; 
    	int * clauseSizes; 
    	int * numOccurrenceT;
    	int maxNumOccurences = 0;
    	int * occurrence;
    	int * occurenceCounter;

    	// engine vars:
    	int global_best_thread;
    	double dmm_alpha = 5.0;
    	double dmm_beta = 20.0;
    	double dmm_gamma = 0.25;
    	double dmm_delta = 0.05;
    	double dmm_epsilon = 0.1;
    	double dmm_zeta = 0.1;
    	double timeout;
    	double init_dt = 1e-8;
    	int seed;
    	int xl_max;
    	int maxsteps = INT_MAX;
    	int tune = 0; //TBC
    	int heuristics = 0; //TBC
    	int digits = 15;
    	volatile int running_threads = 0;
    	char LOC_FILE[1024];
    	std::string SOLUTION_FILE;
    	char PROOF_OF_WORK_FILE[1024];
    	double *v_best;
    	int *loc_thread;
    	int *unit_clause_vars;
    	double *energy_thread;
    	double *energy_thread_min;
    	int *global_thread;
    	int *global_all_runs_thread;
    	double *time_thread;
    	double *time_thread_actual;
    	double *walltime_thread;
    	double *t_init;
    	double *t_begin_thread;
    	double *t_end_thread;
    	int *stepcounter;
    	double *initial_assignments;
    	double *thread_params;
    	double *partable;
    	double *defaults;
    	bool unit_clauses=false;
    	unsigned long long int steps_per_update;
    	struct node {
		    int id;                 //chip-id
		    int *model;             //current assignment
		    int *temporal;          //temp assignment for oracle (not used in production)
		    int *optimal;           //best assignment and solution afterwards
		};
		int solved;
        int global;    
        double global_energy;
        

	    void operator()(int chip_id, int _thread_count, std::atomic<bool>& dynex_quit_flag, std::string input_file, double dmm_alpha, double dmm_beta, double dmm_gamma, double dmm_delta, double dmm_epsilon, double dmm_zeta, double init_dt, bool _dynex_debugger, unsigned long long int _steps_per_update)
	    {
	    	steps_per_update = _steps_per_update;
	    	job_input_file = input_file;
	    	// START WORK:
	    	dynex_debugger = _dynex_debugger;
			bool workfinished = dynex_work(chip_id, _thread_count , dynex_quit_flag, input_file.c_str(), dmm_alpha, dmm_beta, dmm_gamma, dmm_delta, dmm_epsilon, dmm_zeta, init_dt);
			
	    	return;
	    }
	private:
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// FTP FUNCTIONS
		///////////////////////////////////////////////////////////////////////////////////////////////////////////
		#define REMOTE_URL "ftp://ftp.dynexcoin.org/"
		#define FTPUSER "dynexjobs@dynexcoin.org:6`+D6r3:jw1%"

		static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *stream)
		{
		  unsigned long nread;
		  /* in real-world cases, this would probably get this data differently
		     as this fread() stuff is exactly what the library already would do
		     by default internally */
		  //size_t retcode = fread(ptr, size, nmemb, stream);
		  std::size_t retcode = std::fread( ptr, size, nmemb, static_cast<std::FILE*>(stream) );
		 
		  if(retcode > 0) {
		    nread = (unsigned long)retcode;
		    fprintf(stderr, "*** We read %lu bytes from file\n", nread);
		  }
		 
		  return retcode;
		}

		bool upload_file(const std::string filename) {
		
			  std::cout << "c SOLUTION FILE = " << filename << std::endl;
			
			  CURL *curl;
			  CURLcode res;
			  FILE *hd_src;
			  struct stat file_info;
			  unsigned long fsize;
			 
			  /* get the file size of the local file */
			  if(stat(filename.c_str(), &file_info)) {
			    printf("c ERROR: Couldn't open '%s': %s\n", filename.c_str(), strerror(errno));
			    return false;
			  }
			  fsize = (unsigned long)file_info.st_size;
			 
			  printf("c Local file size: %lu bytes.\n", fsize);
			 
			  /* get a FILE * of the same file */
			  hd_src = fopen(filename.c_str(), "rb");
			 
			  /* In windows, this will init the winsock stuff */
			  curl_global_init(CURL_GLOBAL_ALL);
			 
			  /* get a curl handle */
			  curl = curl_easy_init();
			  if(curl) {
			    /* we want to use our own read function */
			    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
			 
			    /* enable uploading */
			    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
			 
			    /* specify target */
			    std::string remoteurl = REMOTE_URL + filename;
			    curl_easy_setopt(curl, CURLOPT_URL, remoteurl.c_str());
			    curl_easy_setopt(curl, CURLOPT_USERPWD, FTPUSER);
			    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
			    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
			    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
			    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
			 
			    /* now specify which file to upload */
			    curl_easy_setopt(curl, CURLOPT_READDATA, hd_src);
			    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)fsize);
			 
			    /* Now run off and do what you have been told! */
			    res = curl_easy_perform(curl);
			    /* Check for errors */
			    if(res != CURLE_OK)
			      fprintf(stderr, "c ERROR: updload failed: %s\n",
				      curl_easy_strerror(res));
			    /* always cleanup */
			    curl_easy_cleanup(curl);
			  }
			  fclose(hd_src); /* close the local file */
			 
			  curl_global_cleanup();
			  return true;
		}

		// work: ------------------------------------------------------------------------------------------------------------------------
		// allocates memory, sets params and initial assignments, then starts the ode integration
		bool dynex_work(int chip_id, int _thread_count, std::atomic<bool>& dynex_quit_flag, const char* input_file, double _dmm_alpha, double _dmm_beta, double _dmm_gamma, double _dmm_delta, double _dmm_epsilon, double _dmm_zeta, double _init_dt) {
			//LogTS << TEXT_CYAN << "[DYNEX CHIP "<<chip_id<<"] STARTING WORK " << job_user_id << "_" << job_id << "..." << TEXT_DEFAULT << std::endl;
			load_input_file(chip_id, input_file);

			dmm_alpha = _dmm_alpha;
			dmm_beta = _dmm_beta;
			dmm_gamma = _dmm_gamma;
			dmm_delta = _dmm_delta;
			dmm_epsilon = _dmm_epsilon;
			dmm_zeta = _dmm_zeta;
			init_dt = _init_dt;
			
			
			/// initialize arrays:
		    v_best = (double *) calloc((size_t) (n+m*2), sizeof(double));
		    loc_thread = (int *) calloc((size_t) 1, sizeof(int));
		    unit_clause_vars = (int *) calloc((size_t) n+1, sizeof(int));
		    energy_thread = (double *) calloc((size_t) 1, sizeof(double));
		    energy_thread_min = (double *) calloc((size_t) 1, sizeof(double));
		    global_thread = (int *) calloc((size_t) 1, sizeof(int));
		    global_all_runs_thread = (int *) calloc((size_t) 1, sizeof(int));
		    time_thread = (double *) calloc((size_t) 1, sizeof(double));
		    time_thread_actual = (double *) calloc((size_t) 1, sizeof(double));
		    walltime_thread = (double *) calloc((size_t) 1, sizeof(double));
		    t_init = (double *) calloc((size_t) 1, sizeof(double));
		    t_begin_thread = (double *) calloc((size_t) 1, sizeof(double));
		    t_end_thread = (double *) calloc((size_t) 1, sizeof(double));
		    stepcounter = (int *) calloc((size_t) 1, sizeof(int));
		    initial_assignments = (double *) calloc((size_t) (n+m*2), sizeof(double));
		    thread_params = (double *) calloc((size_t) PARAM_COUNT, sizeof(double));
		    partable = (double *) calloc((size_t) 4, sizeof(double));
		    defaults = (double *) calloc((size_t) 128, sizeof(double));
			/// detect (and assign) unit clauses (clauses with one literal): ----------------------------
		    int cnt_unit_clauses = 0;
		    for (int i=0; i<m; i++) {
		        if (clauseSizes[i]==1) {
		            cnt_unit_clauses++;
		            unit_clauses = true;
		            int lit = cls[i*MAX_LITS_SYSTEM];
		            if (lit>0) unit_clause_vars[abs(lit)] = 1;
		            if (lit<0) unit_clause_vars[abs(lit)] = -1;
		        }
		    }
		    if (dynex_debugger) printf("c %d UNIT CLAUSES DETECTED.\n",cnt_unit_clauses);
		    /// set _all_runs vars: ---------------------------------------------------------------------
		    global_all_runs_thread[0] = m;
		    /// set t_init: -----------------------------------------------------------------------------
		    t_init[0] = 0.0;
		    /// load assignment: ------------------------------------------------------------------------
		    // TBD
		    /// load partable: --------------------------------------------------------------------------
		    // TBD
		    /// OUPUT SETTINGS: -------------------------------------------------------------------------
		    
		    if (dynex_debugger) {
		        printf(TEXT_CYAN);
		        printf("c [%d] SETTINGS:\n",chip_id);
		        printf("c [%d] MAX STEPS       : ",chip_id); std::cout << maxsteps << std::endl;
		        printf("c [%d] TIMEOUT         : ",chip_id); std::cout << timeout << std::endl;
		        printf("c [%d] INITAL dt       : ",chip_id); std::cout << init_dt << std::endl;
		        printf("c [%d] HEURISTICS      : %d\n",chip_id,heuristics);
		        printf("c [%d] TUNE CIRCUIT    : %d\n",chip_id,tune);
		        printf("c [%d] ALPHA           : ",chip_id); std::cout << dmm_alpha << std::endl; // %.17f\n",dmm_alpha);
		        printf("c [%d] BETA            : ",chip_id); std::cout << dmm_beta << std::endl;
		        printf("c [%d] GAMMA           : ",chip_id); std::cout << dmm_gamma << std::endl;
		        printf("c [%d] DELTA           : ",chip_id); std::cout << dmm_delta << std::endl;
		        printf("c [%d] EPSILON         : ",chip_id); std::cout << dmm_epsilon << std::endl;
		        printf("c [%d] ZETA            : ",chip_id); std::cout << dmm_zeta << std::endl;
		        printf("c [%d] SEED            : %d\n",chip_id,seed);
		        printf("c [%d] XL_MAX          : %.d\n",chip_id,xl_max);

		        printf(TEXT_DEFAULT);
		    }
		    /// init thread specific ODE parameters --------------------------------------------------------
		    /* RNG */
		    std::random_device rd;
			std::seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
		    std::mt19937 generator(sd); 
		    std::uniform_real_distribution<double>  rand_v_double(-1.0, 1.0); // voltage continous between -1 and +1 
		    std::uniform_int_distribution<int>      rand_v(-1, 1); // voltage -1, 0 or +1
		    std::uniform_real_distribution<double>  rand_Xs(0.0, 1.0);
		    std::uniform_real_distribution<double>  rand_Xl(1.0, 100.0); 
		    
	        thread_params[0]    = dmm_alpha;
	        thread_params[1]    = dmm_beta;
	        thread_params[2]    = dmm_gamma;
	        thread_params[3]    = dmm_delta;
	        thread_params[4]    = dmm_epsilon;
	        thread_params[5]    = dmm_zeta;
	        thread_params[6]    = 0.0;
	        thread_params[7]    = 0.0;
	        thread_params[8]    = init_dt;  
		    //initial assignments:
            for (int j=0; j<n; j++) {
            	initial_assignments[j] = rand_v(generator);
            	if (chip_id==0) initial_assignments[j] = 0;  // check this in the light of decentralisation!!
            	if (chip_id==1) initial_assignments[j] = 1;  // check this in the light of decentralisation!!
            	if (chip_id==2) initial_assignments[j] = -1; // check this in the light of decentralisation!!
            }
            for (int j=n; j<n+m; j++) initial_assignments[j] =  0.0; //double(0.0); //rand_Xs(generator); //; 
            for (int j=n+m; j<n+m*2; j++) initial_assignments[j] = 1.0; //double(1.0); //rand_Xl(generator); // //double(1.0); //
	        // override unit clauses for initial assignments:
	        if (unit_clauses) {
	            for (int j=0; j<n; j++) {
	                if (unit_clause_vars[j+1]!=0) initial_assignments[j] = unit_clause_vars[j+1];
	            }
	        }
		    
		    /// prepare SOLUTION file ------------------------------------------------------------------------
		    SOLUTION_FILE = job_input_file + ".solution.txt";
		    
		    /// run solver -----------------------------------------------------------------------------------
            apply(chip_id, dynex_quit_flag);
            
			return true;
		}

		// ---------------------------------------------------------------------------------------------------------------
		// DXDT CALCULATION:
		// ---------------------------------------------------------------------------------------------------------------
		std::vector<double> dmm_generate_dxdt(int chip_id, std::vector<double> x , double t ) {
			
			//timers:
        	t_end_thread[0] = clock();
			double time_spent = (double)(t_end_thread[0] - t_begin_thread[0])/1000;
        	time_thread_actual[0] = t;
        	energy_thread[0] = 0.0;
        	//vectors:
        	std::vector<double> dxdt(n+m*2); // dxdt vector
        	std::vector< std::vector <double >> dxdt_v(n); // vector for storing all voltages
        	/* main ODE routine */
	        if (solved!=1) {
	            int loc;
	            // Loop ---------------------------------------------------------------------------------------------------------------
	            loc = m;
	            // loop through each clause:
	            for (int clause = 0; clause < m; clause++) {
	                // number of literals in this clause:
	                int ksat = clauseSizes[clause];
	                // Xl & Xs:
	                double Xs = x[clause+n];   if (Xs<0.0) Xs = double(0.0); if (Xs>1.0) Xs = double(1.0); //Xs bounds
	                double Xl = x[clause+n+m]; if (Xl<1.0) Xl = double(1.0); if (Xl>xl_max) Xl = double(xl_max); //Xl bounds
	                double C  = double(0.0);
	                double Ri, Rj, Rk, Gi, Gj, Gk;
	                // 3-sat:
	                if (ksat==3) {
	                    int Qi = (cls[clause*MAX_LITS_SYSTEM+0]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                    int Qj = (cls[clause*MAX_LITS_SYSTEM+1]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                    int Qk = (cls[clause*MAX_LITS_SYSTEM+2]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                    int liti = abs(cls[clause*MAX_LITS_SYSTEM+0]);
	                    int litj = abs(cls[clause*MAX_LITS_SYSTEM+1]);
	                    int litk = abs(cls[clause*MAX_LITS_SYSTEM+2]);
	                    double Vi = x[liti-1]; if (Vi<-1.0) Vi = -1.0; if (Vi>1.0) Vi = 1.0; //V bounds
	                    double Vj = x[litj-1]; if (Vj<-1.0) Vj = -1.0; if (Vj>1.0) Vj = 1.0; //V bounds
	                    double Vk = x[litk-1]; if (Vk<-1.0) Vk = -1.0; if (Vk>1.0) Vk = 1.0; //V bounds
	                    double i = double(1.0)-double(Qi)*Vi;
	                    double j = double(1.0)-double(Qj)*Vj;
	                    double k = double(1.0)-double(Qk)*Vk;
	                    C = double(fmin(i, fmin(j, k)));
	                    C = C / double(2.0);
	                    if (C<0.0) C=double(0.0);
	                    if (C>1.0) C=double(1.0);
	                    
	                    // equation Gn,m(vn,vj,vk)= 1/2 qn,mmin[(1−qj,mvj),(1−qk,mvk)] (5.x):
	                    Gi = double(Qi) * fmin(j,k) / double(2.0);
	                    Gj = double(Qj) * fmin(i,k) / double(2.0);
	                    Gk = double(Qk) * fmin(i,j) / double(2.0);
	                    
	                    // equation Rn,m (vn , vj , vk ) = 1/2(qn,m −vn), Cm(vn,vj,vk)= 1/2(1−qn,mvn), 0 otherwise (5.x):
	                    if (C == double(i/double(2.0)) ) {Ri = (double(Qi)-Vi)/2.0;} else {Ri = double(0.0);} //Qi*i/2.0*-1;} //= 0.0
	                    if (C == double(j/double(2.0)) ) {Rj = (double(Qj)-Vj)/2.0;} else {Rj = double(0.0);} //Qj*j/2.0*-1;} //= 0.0
	                    if (C == double(k/double(2.0)) ) {Rk = (double(Qk)-Vk)/2.0;} else {Rk = double(0.0);} //Qk*k/2.0*-1;} //= 0.0
	                    
	                    // equation Vn = SUM xl,mxs,mGn,m + (1 + ζxl,m)(1 − xs,m)Rn,m (5.x):
	                    double _Vi = Xl * Xs * Gi + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * Ri ;
	                    double _Vj = Xl * Xs * Gj + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * Rj ;
	                    double _Vk = Xl * Xs * Gk + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * Rk ;

	                    //sum of vectors method:
	                    if (_Vi!=0.0) dxdt_v[liti-1].push_back(_Vi);
	                    if (_Vj!=0.0) dxdt_v[litj-1].push_back(_Vj);
	                    if (_Vk!=0.0) dxdt_v[litk-1].push_back(_Vk);

	                    // do not change unit_clauses:
	                    if (unit_clauses) {
	                        if (unit_clause_vars[liti]!=0) dxdt[liti-1] = 0.0;
	                        if (unit_clause_vars[litj]!=0) dxdt[litj-1] = 0.0;
	                        if (unit_clause_vars[litk]!=0) dxdt[litk-1] = 0.0;
	                    }
	                }
	                // 2-sat:
	                if (ksat==2) {
	                    int Qi = (cls[clause*MAX_LITS_SYSTEM+0]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                    int Qj = (cls[clause*MAX_LITS_SYSTEM+1]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                    int liti = abs(cls[clause*MAX_LITS_SYSTEM+0]);
	                    int litj = abs(cls[clause*MAX_LITS_SYSTEM+1]);
	                    double Vi = x[liti-1]; if (Vi<-1.0) Vi = -1.0; if (Vi>1.0) Vi = 1.0; 
	                    double Vj = x[litj-1]; if (Vj<-1.0) Vj = -1.0; if (Vj>1.0) Vj = 1.0;
	                    double i = double(1.0)-double(Qi)*Vi;
	                    double j = double(1.0)-double(Qj)*Vj;
	                    C = double(fmin(i, j));
	                    C = C / double(2.0) ;
	                    if (C<0.0) C=double(0.0);
	                    if (C>1.0) C=double(1.0);
	                    //voltage:
	                    Gi = double(Qi) * j / double(2.0);
	                    Gj = double(Qj) * i / double(2.0);
	                    
	                    if (C == double(i/double(2.0)) ) {Ri = (double(Qi)-Vi)/2.0;} else {Ri = double(0.0);} //Qi*i/2.0*-1;} //= 0.0
	                    if (C == double(j/double(2.0)) ) {Rj = (double(Qj)-Vj)/2.0;} else {Rj = double(0.0);} //Qj*j/2.0*-1;} //= 0.0

	                    double _Vi = Xl * Xs * Gi + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * Ri;
	                    double _Vj = Xl * Xs * Gj + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * Rj;

	                    //sum of vectors method:
	                    if (_Vi!=0.0) dxdt_v[liti-1].push_back(_Vi);
	                    if (_Vj!=0.0) dxdt_v[litj-1].push_back(_Vj);
	                    
	                    // do not change unit_clauses:
	                    if (unit_clauses) {
	                        if (unit_clause_vars[liti]!=0) dxdt[liti-1] = 0.0;
	                        if (unit_clause_vars[litj]!=0) dxdt[litj-1] = 0.0;
	                    }
	                }
	                
	                // k-sat:
	                if (ksat!=1 && ksat!=2 && ksat!=3) {
	                    int lit[MAX_LITS_SYSTEM], Q[MAX_LITS_SYSTEM];
	                    double V[MAX_LITS_SYSTEM], _i[MAX_LITS_SYSTEM], R[MAX_LITS_SYSTEM], G[MAX_LITS_SYSTEM];

	                    double c_min=INT_MAX;
	                    for (int i=0; i<ksat; i++) {
	                        Q[i] = (cls[clause*MAX_LITS_SYSTEM+i]>0)? 1:-1; // +1 if literal is >0, otherwise -1
	                        lit[i] = abs(cls[clause*MAX_LITS_SYSTEM+i]);
	                        V[i] = x[lit[i]-1]; if (V[i]<-1.0) V[i]=-1.0; if (V[i]>1.0) V[i]=1.0; //boundary for v € [-1,1]:
	                        _i[i] = double(1.0)-double(Q[i])*V[i];
	                        // find min:
	                        if (_i[i]<c_min) c_min = _i[i]; 
	                    }
	                    C = c_min / double(2.0);
	                    if (C<0.0) printf("*\n");//C=0.0; // never triggered?
	                    if (C>1.0) printf("*\n");//C=1.0; // never triggered?
	                    
	                    for (int i=0; i<ksat; i++) {
	                        //find min of others:
	                        double g_min = INT_MAX;
	                        for (int ii=0; ii<ksat; ii++) {if (ii!=i && _i[ii]<g_min) g_min = _i[ii];}
	                        G[i] = double(Q[i]) * g_min / double(2.0);
	                        double comp = _i[i]/double(2.0);
	                        if (comp<0.0) printf("*\n");//comp = 0.0; // never triggered?
	                        if (comp>1.0) printf("*\n");//comp = 1.0; // never triggered?
	                        if (C != comp) {R[i] = double(0.0);} else {R[i] = (double(Q[i])-V[i]) / double(2.0);}
	                        double _V = Xl * Xs * G[i] + (double(1.0) + dmm_zeta * Xl) * (double(1.0) - Xs) * R[i];
	                        
	                        //sum of vectors method:
	                        if (_V!=0.0) dxdt_v[lit[i]-1].push_back(_V);
	                    
	                        // do not change unit_clauses:
	                        if (unit_clauses) {
	                            if (unit_clause_vars[lit[i]]!=0) dxdt[lit[i]-1] = 0.0;
	                        }
	                    }
	                }

	                //update energy:
	                energy_thread[0] += C;
	                //update loc:
	                if (C<0.5) loc--; //this clause is sat, reduce loc
	                
	                // Calculate new Xs:
	                dxdt[n+clause] = dmm_beta * (Xs + dmm_epsilon) * (C - dmm_gamma);
	                
	                // Calculate new Xl:
	                dxdt[n+m+clause] = dmm_alpha * (C - dmm_delta);
	                
	            } //---clause calculation loop

	            // summation of voltages SUM dxdt_v[n] => dxdt[n]: ------------------------------------------------------
	            for (int i=0; i<n; i++) {
	                std::sort(dxdt_v[i].begin(), dxdt_v[i].end()); //summing with smallest first increases accuracy
	                dxdt[i] = accumulate(dxdt_v[i].begin(), dxdt_v[i].end(), (double) 0.0);
	            }

	            // update global_all_runs_thread & update v_best: --------------------------------------------------------
	            if (loc <= global_all_runs_thread[0]) {
	                global_all_runs_thread[0] = loc;
	                t_init[0] = t;
	                //update v_best array:
	                for (int i=0; i<n+m*2; i++) v_best[i] = x[i]; 
	            }
	            // update loc of thread: ---------------------------------------------------------------------------------
	            loc_thread[0] = loc;
	            time_thread_actual[0] = t;

	            //new lower lock (global)? or lower energy (global)? -----------------------------------------------------
	            if (loc<global || energy_thread[0]<global_energy) {
	                if (loc<global) {
	                    global = loc;
	                    global_best_thread = 0;
	                }
	                if (energy_thread[0]<global_energy) {
	                    global_energy = energy_thread[0];
	                    global_best_thread = 0;
	                }
	                if (energy_thread[0]<energy_thread_min[0]) {
	                    global_best_thread = 0;
	                }
	                if (loc<global_thread[0]) {
	                    global_best_thread = 0;
	                }


	                if (dynex_debugger) {
	                    std::cout << "\rc [" << chip_id << "] " << time_spent/1000 << "s "
	                    << "T=" << t
	                    << " GLOBAL=" << global
	                    << " (LOC=" << loc << ")" 
	                    << " (" << stepcounter[0] << ")"
	                    << " α=" << dmm_alpha
	                    << " β=" << dmm_beta
	                    << " Xs=" << x[n]
	                    << " Xl=" << x[n+m] 
	                    << " Σe=" << energy_thread[0] << " " << std::endl;
	                    fflush(stdout);
	                } 
					
	                // loc file; -----------------------------------------------------------------------------------------
                    //FILE *floc = fopen(LOC_FILE, "a");
                    //fprintf(floc,"%d;%d,%.5f;%.5f;%d;%.2f\n",chip_id, stepcounter[0], time_spent,t,global,global_energy);
                    //fclose(floc);
	            }

	            // update energy of thread: ------------------------------------------------------------------------------
	            if (energy_thread[0] < energy_thread_min[0]) {
	                energy_thread_min[0] = energy_thread[0];
	            }

	            //new thread global? --------------------------------------------------------------------------------------
	            if (loc<global_thread[0]) {
	                // update global_thread, time_thread and walltime_thread:
	                global_thread[0] = loc;
	                time_thread[0] = t;
	                walltime_thread[0] = time_spent;
	            }
	            
	            //solved? -------------------------------------------------------------------------------------------------
	            if (loc==0) {
	            	
		            printf("\nc [CPU %d] T=",chip_id); std::cout << t << TEXT_YELLOW << " SOLUTION FOUND" << std::endl; 
	                for (int i=0; i<n; i++) {
	                    if (x[i]>0) printf("%d ",(i+1));
	                    if (x[i]<0) printf("%d ",(i+1)*-1);
	                }
	                fflush(stdout);
                    printf("\nc [CPU %d] VERIFYING...\n",chip_id); if (dynex_debugger) printf(TEXT_DEFAULT);
                	
                    
                    bool sat = true; bool clausesat;
                    for (int i=0; i<m; i++) {
                        for (int j=0; j<clauseSizes[i]; j++) {
                            clausesat = false;
                            int lit = abs(cls[i*MAX_LITS_SYSTEM+j]);
                            if ( (x[lit-1]>0 && cls[i*MAX_LITS_SYSTEM+j]>0) || (x[lit-1]<0 && cls[i*MAX_LITS_SYSTEM+j]<0) ) {
                                clausesat = true;
                                break;
                            }
                        }
                        if (!clausesat) {
                            sat = false;
                            break;
                        }
                    }

                    printf(TEXT_YELLOW);
                    if (sat)  {
                        printf(TEXT_YELLOW); 
                        printf("c [CPU %d] SAT (VERIFIED)\n",chip_id); 
                        solved = 1;
                        LogTS << "[CHIP" << chip_id << "] " << TEXT_YELLOW << "FOUND A SOLUTION." << TEXT_DEFAULT << std::endl;
                        //write solution to file:
                        FILE *fs = fopen(SOLUTION_FILE.c_str(), "w");
                        for (int i=0; i<n; i++) {
                            if (x[i]>=0) fprintf(fs,"%d, ",i+1);
                            if (x[i]<0) fprintf(fs,"%d, ",(i+1)*-1);
                        }
                        fclose(fs);    
                        //submit solution:
                        upload_file(SOLUTION_FILE);
                        printf("c [CPU %d] SOLUTION UPLOADED TO DYNEX\n",chip_id); 
                    }
                    if (!sat) {
                    	printf(TEXT_RED); 
                    	printf("c [CPU %d] UNSAT (VERIFIED)\n",chip_id);
                    }
                    printf(TEXT_DEFAULT);
	                

	                // update locfile:
                    //FILE *floc = fopen(LOC_FILE, "a");
                    //fprintf(floc,"%d;%d,%.5f;%.5f;%d;%.2f\n",chip_id, stepcounter[0], time_spent,t,global,global_energy);
                    //fclose(floc);
            
	            } // ---output
	        }


		    return dxdt;
		}

		// ---------------------------------------------------------------------------------------------------------------
		// ODE INTEGRATION STEP:
		// ---------------------------------------------------------------------------------------------------------------
		std::vector<double> dmm_dostep(int chip_id, std::vector<double> _x, std::vector<double> dxdt, double h) {

		    //update V:
		    for (int i=0; i<n; i++) {
		        _x[i] = _x[i] + h * dxdt[i];
		        if (unit_clause_vars[i+1]!=0) _x[i] = unit_clause_vars[i+1]; //TODO: assign +1 or -1
		        if (_x[i]<-1.0) _x[i]=-1.0;   
		        if (_x[i]>1.0) _x[i]=1.0;
		    }
		    //update XS:
		    for (int i=n; i<n+m; i++) {
		        _x[i] = _x[i] + h * dxdt[i];
		        if (_x[i]<0.0) _x[i]=0.0;
		        if (_x[i]>1.0) _x[i]=1.0;
		    }
		    //update Xl:
		    for (int i=n+m; i<n+m*2; i++) {
		        _x[i] = _x[i] + h * dxdt[i];
		        if (_x[i]<1.0) _x[i]=1.0;  
		        if (_x[i]>xl_max) _x[i]=xl_max;
		    }
		    
		    // increase stepcounter for this thread:
		    stepcounter[0]++;

		    return _x;
		}

		// dynex ode integration --------------------------------------------------------------------------------------------------------
		int dmm(struct node *node, std::atomic<bool>& dynex_quit_flag) {
			if (dynex_debugger) printf("c [CPU %d] STARTING ODE...\n",node->id);
    		solved = 0;
	        global = m;    
	        global_energy = m;
	        stepcounter[0] = 0;
	        global_thread[0] = m;
			time_thread[0] = 0.0;
			walltime_thread[0] = 0.0;
			energy_thread[0] = m;
			energy_thread_min[0] = m;

		//defining vector: -------------------------------------------------------------------------------
		int size_of_vector = n+m*2;
		state_type x(size_of_vector);

		// initial conditions: ---------------------------------------------------------------------------
		for (int i=0; i<n+m*2; i++) x[i] = initial_assignments[i]; 
		if (dynex_debugger) {
		std::cout << TEXT_DEFAULT << "c [CPU " << node->id << "] INITIAL ASSIGNMENTS SET: "
		<< x[0] << " "
		<< x[1] << " "
		<< x[2] << " "
		<< x[3] << " "
		<< x[4] << " "
		<< x[5] << " "
		<< x[6] << " "
		<< x[7] << " "
		<< x[8] << " "
		<< x[9] << " "
		<< TEXT_DEFAULT << std::endl;
		fflush(stdout);
		}

		//timers: ----------------------------------------------------------------------------------------
		t_begin_thread[0] = clock(); 

    		// ODE integration: ------------------------------------------------------------------------------
    		if (dynex_debugger) printf("c [CPU %d] ADAPTIVE TIME STEP INTEGRATION...\n",node->id);
    		double t = 0.0; // start time
        	double h = init_dt; //thread_params[8]; //init stepsize adaptive
			double h_min = 0.0078125; //0.000125; //
			//double h_max = 10;

        	// run until exit: -------------------------------------------------------------------------------
        	while (solved!=1 && !dynex_quit_flag) {
        		//max steps reached?
        		if (stepcounter[0]>maxsteps) {
        			LogTS << "[CPU " << node->id << "] MAX "<<stepcounter[0]<<" INTEGRATION STEPS REACHED - WE QUIT. " << std::endl;
        			return 0;
        		}
        		// status update?
        		if (stepcounter[0] >0 && stepcounter[0] % steps_per_update == 0) { //10000
        			// screen update:
        			LogTS << TEXT_CYAN << "[CPU " << node->id << "] SIMULATED TIME="<< t <<" ENERGY=" << energy_thread_min[0] << " | GLOBAL=" << global_thread[0] << TEXT_DEFAULT << std::endl;
        		}
				auto dxdt = dmm_generate_dxdt(node->id, x, t);
				
				// adaptive step:
        		bool adaptive_accepted = false;
				auto x_tmp = x;
                int varchanges2;
				h = init_dt; // <=== CHECK IF WE NEED THIS
				
                while (!adaptive_accepted) {
					x_tmp = dmm_dostep(node->id, x, dxdt, h);
					varchanges2 = 0; // # vars changing by x %
					for (int i=0; i<n; i++) {
						if (fabs(x_tmp[i]-x[i])>=1) varchanges2++; // maximum allowed voltage change
					}
					if (varchanges2>0) {h = h * 1/2; } else {adaptive_accepted = true;}
					stepcounter[0]--;
					if (h<=h_min) adaptive_accepted = true;
                }
				
                x = x_tmp;
                stepcounter[0]++;
                t = t + h; // increase ODE time by stepsize

				

	            // solved? if yes, we are done
	            if (solved) {
					//move solution to node:
	                for (int i=0; i<n; i++) {
	                    if (x[i]>0) node->optimal[i] = 1;
	                    if (x[i]<0) node->optimal[i] = 0;
	                }
	                return 1;
	            }

        	}
			//---

			return 0;
		}

		// dynex solver sequence: -------------------------------------------------------------------------------------------------------
		void apply(int chip_id, std::atomic<bool>& dynex_quit_flag) {
		    
		    struct node node;
		    node.id = chip_id;
		    node.temporal = (int *) calloc((size_t) n, sizeof(int));
		    node.model = (int *) calloc((size_t) n, sizeof(int));
		    node.optimal = (int *) calloc((size_t) n, sizeof(int));

		    //run ODE:
		    LogTS << TEXT_CYAN << "[CPU "<<chip_id<<"] STARTING ODE INTEGRATION..." << TEXT_DEFAULT << std::endl;
			int result = dmm(&node, dynex_quit_flag); // <== run ODE integration
			// show time:
    		t_end_thread[0] = clock();
    		double time_spent = (double)(t_end_thread[0] - t_begin_thread[0]) / 1000; 
    		if (dynex_debugger) printf("c [CPU %d] TIME SPENT: %.5fs\n",chip_id,time_spent);

    		// print solution:
		    if (result == 1) {
		        printf("\ns [CPU %d] SATISFIABLE",chip_id);
		        for (int i = 0; i < n; i++) {
		            if (i % 20 == 0) printf("\nv ");
		            printf("%i ", node.optimal[i] ? +(i + 1) : -(i + 1));
		        }
		        printf("0\n");
		        fflush(stdout);
				dynex_quit_flag = true;
		    }
		    if (result == 0) {
		        if (dynex_debugger) printf("\ns [CPU %d] UNKNOWN\n",chip_id);
		    }
		    // free memory:
		    free(node.temporal);
		    free(node.model);
		    free(node.optimal);
		}

		// download input file: ---------------------------------------------------------------------------------------------------------
		// downloads and parses input file, generates arrays for cls, occurences, unit clauses, etc.
		bool load_input_file(int chip_id, const char* job_input_file) {
			
			int i, j;
			char buffer[512];
			if (dynex_debugger) LogTS << TEXT_CYAN << "[CPU "<<chip_id<<"] LOADING INPUT FILE..." << job_input_file << TEXT_DEFAULT << std::endl;
			
			/// load CNF header:
			int ret;
			FILE* file = fopen(job_input_file, "r");
			if (strcmp(buffer, "c") == 0) {
				while (strcmp(buffer, "\n") != 0) {
					ret = fscanf(file, "%s", buffer);
				}
			}
			while (strcmp(buffer, "p") != 0) {
				ret = fscanf(file, "%s", buffer);
			}
			ret = fscanf(file, " cnf %i %i", &n, &m);

			xl_max = xl_max * m;
			if (xl_max <= 0) xl_max = INT_MAX;

			/// reserve  memory - needs to be done before anything else:
			cls = (int*)calloc((size_t)m * MAX_LITS_SYSTEM, sizeof(int));
			for (int i = 0; i < m * MAX_LITS_SYSTEM; i++) cls[i] = 0;
			numOccurrenceT = (int*)calloc((size_t)n + 1, sizeof(int));
			clauseSizes = (int*)calloc((size_t)m, sizeof(int));

			/// read CNF: /////////////////////////////////////////
			int lit;
			for (i = 0; i < m; i++) {
				//if (dynex_debugger) std::cout << "\rc LOADING " << (100.0 * (i + 1) / m) << "%                       ";
				//if (dynex_debugger) fflush(stdout);
				j = 0;
				do {
					ret = fscanf(file, "%s", buffer);
					if (strcmp(buffer, "c") == 0) {
						ret = fscanf(file, "%512[^\n]\n", buffer);
						continue;
					}
					lit = atoi(buffer);
					cls[i * MAX_LITS_SYSTEM + j] = lit;
					// increase number of Occurence of the variable, max number of occurences
					if (lit != 0) {
						numOccurrenceT[abs(lit)]++;
						if (numOccurrenceT[abs(lit)] > maxNumOccurences) { maxNumOccurences = numOccurrenceT[abs(lit)]; }
						clauseSizes[i] = j + 1;
					}
					j++;
				} while (strcmp(buffer, "0") != 0);
				j--;
				if (j > MAX_LITS_SYSTEM) {
					printf("c ERROR: CLAUSE %d HAS MORE THAN %d LITERALS.\n", i, MAX_LITS_SYSTEM);
					return EXIT_FAILURE;
				}
			}
			//if (dynex_debugger) std::cout << std::endl;

			if (dynex_debugger) printf("c MAX VARIABLE OCCURENCE: %'d\n", maxNumOccurences);

			if (dynex_debugger) printf("c FIRST 10 CLAUSES:\n");
			for (i = 0; i < 11; i++) {
				if (dynex_debugger) printf("c CLAUSE %i: ", i);
				for (j = 0; j < clauseSizes[i]; j++) { if (dynex_debugger) printf(" %d", cls[i * MAX_LITS_SYSTEM + j]); }
				if (dynex_debugger) printf(" (%d)", clauseSizes[i]);
				if (dynex_debugger) printf("\n");
			}

			if (dynex_debugger) printf("c LAST 10 CLAUSES:\n");
			for (i = m - 1; i > (m - 10); i--) {
				if (dynex_debugger) printf("c CLAUSE %i: ", i);
				for (j = 0; j < clauseSizes[i]; j++) { if (dynex_debugger) printf(" %d", cls[i * MAX_LITS_SYSTEM + j]); }
				if (dynex_debugger) printf(" (%d)", clauseSizes[i]);
				if (dynex_debugger) printf("\n");
			}

			//build occurence array: [var][cls...] /////////////////////////////////////////
			occurrence = (int*)calloc((size_t)(n + 1) * maxNumOccurences, sizeof(int));
			occurenceCounter = (int*)calloc((size_t)n + 1, sizeof(int));

			for (i = 0; i < m; i++) {
				for (j = 0; j < clauseSizes[i]; j++) {
					lit = abs(cls[i * MAX_LITS_SYSTEM + j]);
					occurrence[lit * maxNumOccurences + occurenceCounter[lit]] = i;
					occurenceCounter[lit]++;
				}
			}

			//output:
			if (dynex_debugger) {
				printf("c OCCURENCES: ");
				for (i = 1; i <= 20; i++) printf("%d->%d ", i, occurenceCounter[i]);
				printf("\n");
			}

			return true;
		}

		// retrieve new work for dynex chips: -------------------------------------------------------------------------------------------
		bool dynex_get_work(int chip_id, int thread_count, uint64_t dynex_minute_rate) {
			// replaced with MALLOB
			return true;
		}
};

//---------------------------------------------------------------------------------------------------------------------------
// dyndex chip class
//---------------------------------------------------------------------------------------------------------------------------
namespace Dynex {

	  class dynexchip {

			  public:
			  	// dynex chip variables:
			  	int 								dynex_chip_threads;
			  	bool 								dynex_chips_running = false; 
			  	int 								dynex_chip_state = DYNEX_STATE_OFF; 
			  	std::atomic_bool 					dynex_quit_flag ;

			  	bool init() {
			    	return true;
			    };

			    bool start(size_t threads_count, std::string _job_input_file, double dmm_alpha, double dmm_beta, double dmm_gamma, double dmm_delta, double dmm_epsilon, double dmm_zeta, double init_dt, bool _dynex_debugger, unsigned long long int steps_per_update) {

				if (!threads_count) return false;

				//already running?
			    	if (dynex_chips_running) {
			    		LogTS << TEXT_CYAN << "CANNOT START DYNEX CHIPS - ALREADY RUNNING" << TEXT_DEFAULT << std::endl;
			    		return false;
			    	}
			    	
			    	// set receiving address, threads, rate, etc:
			    	dynex_chip_threads = threads_count;
			    	dynex_chips_running = true;
			    	dynex_chip_state = DYNEX_STATE_IDLE;
			    	
			    	LogTS << TEXT_CYAN << "[CPU] STARTING " << dynex_chip_threads << " DYNEX CHIPS ON CPU " << TEXT_DEFAULT << std::endl;
			    	dynex_quit_flag = false;
			    	// start chip threads:
			    	for (size_t i=0; i<threads_count; i++) {
			    		std::thread observer_th(dynex_chip_thread_obj(), i, dynex_chip_threads, std::ref(dynex_quit_flag), _job_input_file, dmm_alpha, dmm_beta, dmm_gamma, dmm_delta, dmm_epsilon, dmm_zeta, init_dt, _dynex_debugger, steps_per_update );
				    	observer_th.detach();
						assert(!observer_th.joinable());
						std::this_thread::sleep_for(std::chrono::milliseconds(250));
			    	}
			    	return true;
			    }

			    bool stop() {
			    	dynex_chips_running = false;
			    	dynex_quit_flag = true;
			    	dynex_chip_state = DYNEX_STATE_OFF;
			    	LogTS << TEXT_CYAN << "DYNEX CHIPS STOPPED." << TEXT_DEFAULT << std::endl;
			    	return true;
			    };

			    //bool worker_thread(uint32_t th_local_index);
	  };

}

