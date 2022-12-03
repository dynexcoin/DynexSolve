// ----------------------------------------------------------------------------------------------------
// DYNEXSOLVE
// ----------------------------------------------------------------------------------------------------
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

// example run:
// ./dynexsolve -mining-address XwmfZnrGxfsDsg896z4CLfVzddQiYbMSTWcHoH2XT4c9RHRKHjon6QrAnS9QoSFL9BiQ5cShvnieSMwtMHik2gsF2zjgwL84Y -stratum-url dnx.sg.ekapool.com -stratum-port 19331 -stratum-paymentid 3683cac3c8790d2f35808c7fc11b15f56034961a4497d418f930307346ead71a -stratum-password child@worker -no-cpu
// nvcc ip_sockets.cpp portability_fixes.cpp tcp_sockets.cpp dprintf.cpp jsonxx.cc Dynexchip.cpp kernel.cu -o dynexsolve -O4 -lcurl libCrypto.a

#pragma comment(lib, "libcurl.lib" )
#pragma comment(lib, "winmm.lib" )
#pragma comment(lib, "ws2_32.lib")
#pragma comment (lib, "zlib.lib")
#pragma comment (lib, "advapi32.lib")
#pragma comment (lib, "crypt32.lib")

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <math.h>
#include <stdbool.h>
#include <locale.h>
#include "memory.h"
#include <chrono>

#include <map>
#include <vector>
#include <iostream>
#include <fstream> 

#include <sstream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include <iomanip>

//#define CURL_STATICLIB
#include <curl/curl.h> //required for MPI - dependency
#include "jsonxx.h"    //no install required + cross platform - https://github.com/hjiang/jsonxx

// for ftp:
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// terminal colors, WinApi header
#ifdef WIN32
#include <windows.h>
#include "termcolor.hpp";
using namespace termcolor;
#endif

#include "Dynexchip.cpp"
#include "dynexservice.cpp" 

using namespace std;

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;

std::string VERSION = "2.1.0";

/// init curl:
CURL* curl;

////////////////////////////////////////////////////////////////////////////////////////////////
// Dynex colors
#ifdef WIN32
#define TEXT_DEFAULT  reset
#define TEXT_YELLOW   magenta
#define TEXT_GREEN    green
#define TEXT_RED      red
#define TEXT_BLUE     blue
#define TEXT_CYAN     cyan
#define TEXT_WHITE    white
#define TEXT_SILVER   yellow
#else
#define TEXT_DEFAULT  "\033[0m"
#define TEXT_YELLOW   "\033[1;33m"
#define TEXT_GREEN    "\033[1;32m"
#define TEXT_RED      "\033[1;31m"
#define TEXT_BLUE     "\033[1;34m"
#define TEXT_CYAN     "\033[1;36m"
#define TEXT_WHITE    "\033[1;37m"
#define TEXT_SILVER   "\033[1;315m"
#endif 

////////////////////////////////////////////////////////////////////////////////////////////////
#define JOB_STATUS_INTRODUCED 			0
#define JOB_STATUS_RUNNING 			1
#define JOB_STATUS_FINISHED_SOLVED 		2
#define JOB_STATUS_FINISHED_UNKNOWN 		3
#define JOB_STATUS_CANCELLED 			4

#define JOB_TYPE_SAT 				0
#define JOB_TYPE_MILP				1
#define JOB_TYPE_QUBO				2
#define JOB_TYPE_MAXSAT				3
#define JOB_TYPE_FEDERATED_ML			4
#define JOB_TYPE_PRETRAINING_ML			5
#define JOB_TYPE_SUBSET_SUM			6
#define JOB_TYPE_INTEGER_FACTORISATION		7

#define ATOMIC_STATUS_ASSIGNED			0
#define ATOMIC_STATUS_RUNNING			1
#define ATOMIC_STATUS_FINISHED_SOLVED		2
#define ATOMIC_STATUS_FINISHED_UNKNOWN		3
#define ATOMIC_STATUS_INTERRUPTED		4

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// global variables:
int n; 				// number of variables
int m; 				// number of clauses
int* a; 			// first row of input formulation
int* b; 			// second row of input formulation
int* c; 			// third row of input formulation
int* adj_opp; 			// adjancences of all opposite literals
int* adj_opp_sizes; 		// adjancences of all opposite literals (sizes)
bool* solution; 	  	// will contain the solution if found (size = n)

std::string JOB_FILENAME; 
int * CHIP_FROM;
int * CHIP_TO;
uint64_cu *h_total_steps_init;
int * jobs_bytes;
int * num_jobs;
uint64_cu* h_total_steps;

// job data structure:
typedef struct
{
    int threadi;

    int* lambda;
    int lambda_pos;
    bool* lambda_bin;

    uint64_cu complexity_counter;
    int n;
    int m;
    int stage;
    bool polarity;
    
    int* new_units;
    int new_units_pos;
    bool* new_units_bin;

    bool isSat;
    int Xk;
    int starting_Xk;
    bool* header;
    int max_adj_size;

    bool flipped;

} job_struct;

// cuda vars:
#define MAX_NUM_GPUS 128 //128 maximum number of multi-GPUs
int nDevices; // number of cuda devices
bool use_multi_gpu = false;

int* d_adj_opp[MAX_NUM_GPUS]{}; 							
int* d_adj_opp_sizes[MAX_NUM_GPUS]{};						
int* d_a[MAX_NUM_GPUS]{}; // 1st row of formulation		
int* d_b[MAX_NUM_GPUS]{}; // 2nd row of formulation		
int* d_c[MAX_NUM_GPUS]{}; // 3rd row of formulation		
job_struct* d_jobs[MAX_NUM_GPUS]{};						
job_struct* h_jobs[MAX_NUM_GPUS]{};
 
// system definitions:
int max_lits_system = 3;
int max_adj_size = 0;
uint64_cu PARALLEL_RUNS;
bool debug = false;
bool mallob_debug = false;
bool testing = false;
std::string testing_file;
bool DISC_OPERATION = false;
std::vector<int> disabled_gpus;
auto t1 = std::chrono::high_resolution_clock::now();

// mallob definitions:
bool MALLOB_ACTIVE = false;
std::string mallob_endpoint = "https://dynex.dyndns.org/dynexmallob" ; //"https://dynexmallob.dynexcoin.org"; // "https://dynex.dyndns.org/dynexmallob"; 
int JOB_ID = -1; // undefined at init; JOB_ID is set from mallob
std::string MALLOB_NETWORK_ID;

// blockchain daemon default parameters:
std::string MINING_ADDRESS = ""; //f.e. "XwnV1b9sULyFvmW8NGQyndJGWkF9eE13XKobuGvHUS4QFRrKH7Ze8tRFM6kPeLjLHyfLWPoo7r8RJKyqpcGxZHk32f2avgT4t";
std::string DAEMON_HOST = "localhost";
std::string DAEMON_PORT = "18333";
float rem_hashrate = 0;

// hasher
#define MAX_KH 10000           // 
#define HASHES_PER_SECOND 1000 // 
#define OPTIMAL_TIME 50        // 
#define RUNNING_BONUS 1        // 

// stratum
bool stratum                   = false;
std::string STRATUM_URL        = ""; //f.e. "http://dynex-test-pool.neuropool.net";
std::string STRATUM_PORT       = ""; //f.e. "19333";
std::string STRATUM_PAYMENT_ID = ""; //f.e. "3683cac3c8790d2f35808c7fc11b15f56034961a4497d418f930307346ead71a";
std::string STRATUM_PASSWORD   = ""; //f.e. "child@worker";

// Dynex Services:
Dynex::dynexchip dynexchip;
Dynexservice::dynexservice dynexservice;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// system helper functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// pretty log time
std::string log_time() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
    auto str = oss.str();

    return str;
}

// curl return value function
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp){
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}
// file existing function
inline bool file_exists(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }  
}

std::vector<std::string> split(std::string text, char delim) {
    std::string line;
    std::vector<std::string> vec;
    std::stringstream ss(text);
    while(std::getline(ss, line, delim)) {
        vec.push_back(line);
    }
    return vec;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// BLOCK CHAIN FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// USED ONLY TO TEST CONNECTION TO DAEMON//
bool stop_miner() {
	std::string url = "http://" + DAEMON_HOST + ":" + DAEMON_PORT + "/stop_mining";
	struct curl_slist *list = NULL; //header list
	list = curl_slist_append(list, "content-type:application/json;");
    	list = curl_slist_append(list, "Connection: close");
	CURLcode res;
	std::string readBuffer;
	curl = curl_easy_init();
    	bool success = true;
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L); // 5s
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5 s
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 0); 
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        if(res != CURLE_OK) {
			fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            success = false;
		}
		
	}
	return success;
}

// rpc command handler: -----------------------------------------------------------------------------------
jsonxx::Object invokeJsonRpcCommand(std::string method, std::vector<std::string> params) {
	jsonxx::Object retval;
	
	std::string url = "http://" + DAEMON_HOST + ":" + DAEMON_PORT + "/json_rpc";
	std::string postfields = "{\"jsonrpc\":\"2.0\",\"method\":\""+method+"\",\"params\":{";
	
	if (params.size()>0) {
		for (int i=0; i<params.size(); i++) postfields = postfields+params[i]+",";
		postfields.pop_back();
	}
	
	postfields = postfields + "}}";
	//std::cout << TEXT_GREEN << "postfields: " << postfields << TEXT_DEFAULT << std::endl;
	//CURL *curl;
	CURLcode res;
	std::string readBuffer;
	//curl_global_init(CURL_GLOBAL_ALL);
	curl = curl_easy_init();
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postfields.c_str());
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 0);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L); // 5s
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5 s
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
		if(res != CURLE_OK) {
			fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		} else {
			//std::cout << TEXT_GREEN << "readBuffer: " << readBuffer << TEXT_DEFAULT << std::endl;
			std::istringstream input(readBuffer);
			retval.parse(input);
		}
		
	}
	//curl_global_cleanup();
	return retval;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// FTP FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define REMOTE_URL "ftp://ftp.dynexcoin.org/"
#define FTPUSER "dynexjobs@dynexcoin.org:6`+D6r3:jw1%"

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *stream)
{
  unsigned long nread;
  std::size_t retcode = std::fread( ptr, size, nmemb, static_cast<std::FILE*>(stream) );
 
  if(retcode > 0) {
    nread = (unsigned long)retcode;
    fprintf(stderr, "*** We read %lu bytes from file\n", nread);
  }
 
  return retcode;
}

struct FtpFile {
  const char *filename;
  FILE *stream;
};
 
static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
  struct FtpFile *out = (struct FtpFile *)stream;
  if(!out->stream) {
    /* open file for writing */
    out->stream = fopen(out->filename, "wb");
    if(!out->stream)
      return -1; /* failure, cannot open file to write */
  }
  return fwrite(buffer, size, nmemb, out->stream);
}

bool upload_file(const std::string filename) {
	
	  //CURL *curl;
	  CURLcode res;
	  FILE *hd_src;
	  struct stat file_info;
	  unsigned long fsize;
	 
	  /* get the file size of the local file */
	  if(stat(filename.c_str(), &file_info)) {
	    printf(" [INFO] ERROR: Couldn't open '%s': %s\n", filename.c_str(), strerror(errno));
	    return false;
	  }
	  fsize = (unsigned long)file_info.st_size;
	 
	  printf(" [INFO] Local file size: %lu bytes.\n", fsize);
	 
	  /* get a FILE * of the same file */
	  hd_src = fopen(filename.c_str(), "rb");
	 
	  /* In windows, this will init the winsock stuff */
	  //curl_global_init(CURL_GLOBAL_ALL);
	 
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
	      fprintf(stderr, " [INFO] ERROR: updload failed: %s\n",
		      curl_easy_strerror(res));
	    /* always cleanup */
	    curl_easy_cleanup(curl);
	  }
	  fclose(hd_src); /* close the local file */
	 
	  //curl_global_cleanup();
	  return true;
}

bool download_file(const std::string filename) {
	
	//CURL *curl;
	CURLcode res;
	struct FtpFile ftpfile = {
	filename.c_str(), /* name to store the file as if successful */
	NULL
	};
	
	//curl_global_init(CURL_GLOBAL_DEFAULT);
 
	curl = curl_easy_init();
	if(curl) {
		std::string remoteurl = REMOTE_URL + filename;
		curl_easy_setopt(curl, CURLOPT_URL, remoteurl.c_str());
		curl_easy_setopt(curl, CURLOPT_USERPWD, FTPUSER);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
		curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
		/* Define our callback to get called when there's data to be written */
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
		/* Set a pointer to our struct to pass to the callback */
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpfile);
		/* Switch on full protocol/debug output */
		//curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
		res = curl_easy_perform(curl);
		/* always cleanup */
		curl_easy_cleanup(curl);
		if(CURLE_OK != res) {
			/* we failed */
			fprintf(stderr, " [INFO] ERROR: download failed: %d\n", res);
			return false;
		}
	} else {
		return false;
	}

	if(ftpfile.stream)
	fclose(ftpfile.stream); /* close the local file */

	//curl_global_cleanup();
	
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Mallob Interface
///////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// Mallob MPI command function
// Returns json object
////////////////////////////////////////////////////////////////////////////////////////////////
std::string mallob_follow_url(std::string mallob_endpoint) {

	std::string retval = "";
	std::string readBuffer;

	CURLcode curl_res;
	curl = curl_easy_init();
	if (curl)
	    {
		curl_easy_setopt(curl, CURLOPT_URL, mallob_endpoint.c_str());
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
		curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)"); 
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		curl_res = curl_easy_perform(curl);
		if(curl_res != CURLE_OK)
		  fprintf(stderr, "ERROR during mallob_follow_url: %s\n",
		          curl_easy_strerror(curl_res));

		if(CURLE_OK == curl_res) 
		{
		     char *url;
		     curl_res = curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &url);

		     if((CURLE_OK == curl_res) && url)
		         retval = url;
		         retval.pop_back(); // remove last / from url
		}
		curl_easy_cleanup(curl);
	    }
	    else
	    {
		printf("cURL error.\n");
	    }

    return retval;
}

jsonxx::Object mallob_mpi_command(std::string method, std::vector<std::string> params) {

    jsonxx::Object retval;
    retval.parse("{\"ERROR\":true}");
	std::string url = mallob_endpoint + "/json_rpc.php?method="+method;
	for (int i=0; i<params.size(); i++) url = url + "&" + params[i];
	if (mallob_debug) std::cout << TEXT_GREEN << url << TEXT_DEFAULT << std::endl;
	if (mallob_debug) std::cout << TEXT_GREEN << url.c_str() << TEXT_DEFAULT << std::endl;
	//CURL *curl;
	CURLcode res;
	struct curl_slist *list = NULL; //header list
	std::string readBuffer;
	// header:
	list = curl_slist_append(list, "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9");
	list = curl_slist_append(list, "Accept-Encoding: *");//gzip, deflate, br"); //
	list = curl_slist_append(list, "Accept-Language: en-GB,en-US;q=0.9,en;q=0.8");
	// retrieve:
	curl = curl_easy_init();
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str() );
        	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L); // 5s
        	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L); // 5 s
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
        	curl_easy_cleanup(curl);
		if(res != CURLE_OK) {
			std::cout << log_time() << " [MALLOB] MESSAGE: " << curl_easy_strerror(res) << std::endl;
			retval.parse("{\"ERROR\":true}");
		} else {
			if (mallob_debug) std::cout << TEXT_GREEN << "returns: " << readBuffer << TEXT_DEFAULT << std::endl;
			//curl_easy_cleanup(curl); // moved right after _perform
			readBuffer.erase(std::remove(readBuffer.begin(), readBuffer.end(), '{'), readBuffer.end());
			readBuffer.erase(std::remove(readBuffer.begin(), readBuffer.end(), '}'), readBuffer.end());
			readBuffer.erase(std::remove(readBuffer.begin(), readBuffer.end(), '['), readBuffer.end());
			readBuffer.erase(std::remove(readBuffer.begin(), readBuffer.end(), ']'), readBuffer.end());
			readBuffer = "{" + readBuffer + "}";
			//std::cout << "=> " << readBuffer << std::endl;
			std::istringstream input(readBuffer);
			retval.parse(input);
			if (mallob_debug) std::cout << TEXT_GREEN << "parsed " << retval.json() << TEXT_DEFAULT << std::endl;
		}
        
	} else {
		retval.parse("{\"ERROR\":true}");
	}
	//curl_global_cleanup();
	return retval;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////
bool load_cnf(const char* filename) {
    int i, j;
    char buffer[256];

    std::cout << log_time() << " [INFO] LOADING FILE " << filename << std::endl;
    FILE* file = fopen(filename, "r");
    if (strcmp(buffer, "c") == 0) {
        while (strcmp(buffer, "\n") != 0) {
            fscanf(file, "%s", buffer);
        }
    }
    while (strcmp(buffer, "p") != 0) {
        fscanf(file, "%s", buffer);
    }
    fscanf(file, " cnf %i %i", &n, &m);

    std::cout <<log_time(); printf(" [INFO] VARIABLES : %d\n", n);
    std::cout <<log_time(); printf(" [INFO] CLAUSES   : %d\n", m);
    std::cout <<log_time(); printf(" [INFO] RATIO     : %lf\n", (double)m / n);

    /// reserve  memory:
    int* cls = (int*)calloc((size_t)m * max_lits_system, sizeof(int)); // <=== CAN BE REMOVED OR USED FOR DMM
    a = (int*)calloc((size_t)m, sizeof(int));
    b = (int*)calloc((size_t)m, sizeof(int));
    c = (int*)calloc((size_t)m, sizeof(int));

    // read CNF:
    int lit;
    for (i = 0; i < m; i++) {
        if (i % 10000 == 0) {
		std::cout<<"\r"<<log_time(); printf(" [INFO] LOADING   : %3.2lf%%", 100.0 * (i + 1) / m);
        	fflush(stdout);
        }
	j = 0;
        do {
            fscanf(file, "%s", buffer);
            if (strcmp(buffer, "c") == 0) {
                continue;
            }
            lit = atoi(buffer);
            cls[i * max_lits_system + j] = lit;
            if (j == 0) a[i] = lit;
            if (j == 1) b[i] = lit;
            if (j == 2) c[i] = lit;
            j++;
        } while (strcmp(buffer, "0") != 0);
        j--;
        if (j > max_lits_system) {
            printf(" [INFO] ERROR: CLAUSE %d HAS %d LITERALS (%d ALLOWED).\n", i, j, max_lits_system);
            return false;
        }
        if (j == 2) {
            // add same literal to make it 3sat:
            cls[i * max_lits_system + 2] = cls[i * max_lits_system + 1];
            c[i] = b[i];
            //printf(" [INFO] CLAUSE %d: CONVERTED 2SAT->3SAT: %d %d %d\n", i, cls[i*max_lits_system+0], cls[i*max_lits_system+1], cls[i*max_lits_system+2] ); 
        }
        if (j == 1) {
            // add same literal to make it 3sat:
            cls[i * max_lits_system + 1] = cls[i * max_lits_system + 0];
            cls[i * max_lits_system + 2] = cls[i * max_lits_system + 0];
            b[i] = a[i];
            c[i] = a[i];
            //printf(" [INFO] CLAUSE %d: CONVERTED 1SAT->3SAT: %d %d %d\n", i, cls[i*max_lits_system+0], cls[i*max_lits_system+1], cls[i*max_lits_system+2] ); 
        }
    }
    fclose(file);
    std::cout<<"\r"<<log_time(); printf(" [INFO] LOADING   : %3.2lf%%", 100.0);
    fflush(stdout);
    printf("\n");

    if (debug) {
        printf(" [INFO] FIRST 10 CLAUSES:\n");
        for (i = 0; i < 10; i++) {
            printf(" [INFO] CLAUSE %i: ", i);
            for (j = 0; j < max_lits_system; j++) { printf(" %d", cls[i * max_lits_system + j]); }
            //printf(" CONTROL: %d %d %d",a[i],b[i],c[i]);
            printf("\n");
        }    
    }
    
    /// Build adjances of all oppositve literals:
    // STD::VECTOR here to save 2gb memory limit - OR: MAXIMUM adj_sizes here (and not 2*n+1)
    // ....

    int* adj_sizes = (int*)calloc((size_t)(2 * n + 1), sizeof(int));
    for (j = 1; j <= n * 2; j++) adj_sizes[j] = 0;

    // find max_adj_size (save memory):
    for (int k = 0; k < m; k++) {
        if (a[k] > 0) {
            adj_sizes[a[k]]++; if (adj_sizes[a[k]] > max_adj_size) max_adj_size = adj_sizes[a[k]];
            adj_sizes[a[k]]++; if (adj_sizes[a[k]] > max_adj_size) max_adj_size = adj_sizes[a[k]];
        }
        else {
            adj_sizes[(-a[k] + n)]++; if (adj_sizes[-a[k] + n] > max_adj_size) max_adj_size = adj_sizes[-a[k] + n];
            adj_sizes[(-a[k] + n)]++; if (adj_sizes[-a[k] + n] > max_adj_size) max_adj_size = adj_sizes[-a[k] + n];
        }
        if (b[k] > 0) {
            adj_sizes[b[k]]++; if (adj_sizes[b[k]] > max_adj_size) max_adj_size = adj_sizes[b[k]];
            adj_sizes[b[k]]++; if (adj_sizes[b[k]] > max_adj_size) max_adj_size = adj_sizes[b[k]];
        }
        else {
            adj_sizes[(-b[k] + n)]++; if (adj_sizes[-b[k] + n] > max_adj_size) max_adj_size = adj_sizes[-b[k] + n];
            adj_sizes[(-b[k] + n)]++; if (adj_sizes[-b[k] + n] > max_adj_size) max_adj_size = adj_sizes[-b[k] + n];
        }
        if (c[k] > 0) {
            adj_sizes[c[k]]++; if (adj_sizes[c[k]] > max_adj_size) max_adj_size = adj_sizes[c[k]];
            adj_sizes[c[k]]++; if (adj_sizes[c[k]] > max_adj_size) max_adj_size = adj_sizes[c[k]];
        }
        else {
            adj_sizes[(-c[k] + n)]++; if (adj_sizes[-c[k] + n] > max_adj_size) max_adj_size = adj_sizes[-c[k] + n];
            adj_sizes[(-c[k] + n)]++; if (adj_sizes[-c[k] + n] > max_adj_size) max_adj_size = adj_sizes[-c[k] + n];
        }
    }
    //printf(" [INFO] max_adj_size = %d\n", max_adj_size);
    //printf(" [INFO] max memory for this = %u\n", (2 * n + 1) * max_adj_size);
    for (j = 1; j <= n * 2; j++) adj_sizes[j] = 0;
    //---

    //int * adj = (int *) calloc((size_t) m*(2*n+1), sizeof(int)); // <== needs 3GB! (2GB LIMIT) -> use vector first, then transform to array
    int* adj = (int*)calloc((size_t)(2 * n + 1) * max_adj_size, sizeof(int)); // m with max_adj_size replaced
    std::cout <<log_time();printf(" [INFO] %u BYTES ALLOCATED...\n", (2 * n + 1) * max_adj_size * 4); // int = 4 bytes

    for (int k = 0; k < m; k++) {
        if (k % 1000 == 0) {
		std::cout<<"\r"<<log_time(); printf(" [INFO] BUILDING CIRCUIT(1/2): %d/%d ", k, m);
	}
        if (a[k] > 0) {
            adj[a[k] * max_adj_size + adj_sizes[a[k]]] = b[k]; adj_sizes[a[k]]++; // m with max_adj_size replaced
            adj[a[k] * max_adj_size + adj_sizes[a[k]]] = c[k]; adj_sizes[a[k]]++; // m with max_adj_size replaced
        }
        else {
            adj[(-a[k] + n) * max_adj_size + adj_sizes[(-a[k] + n)]] = b[k]; adj_sizes[(-a[k] + n)]++; // m with max_adj_size replaced
            adj[(-a[k] + n) * max_adj_size + adj_sizes[(-a[k] + n)]] = c[k]; adj_sizes[(-a[k] + n)]++; // m with max_adj_size replaced
        }
        if (b[k] > 0) {
            adj[b[k] * max_adj_size + adj_sizes[b[k]]] = a[k]; adj_sizes[b[k]]++; // m with max_adj_size replaced
            adj[b[k] * max_adj_size + adj_sizes[b[k]]] = c[k]; adj_sizes[b[k]]++; // m with max_adj_size replaced
        }
        else {
            adj[(-b[k] + n) * max_adj_size + adj_sizes[(-b[k] + n)]] = a[k]; adj_sizes[(-b[k] + n)]++; // m with max_adj_size replaced
            adj[(-b[k] + n) * max_adj_size + adj_sizes[(-b[k] + n)]] = c[k]; adj_sizes[(-b[k] + n)]++; // m with max_adj_size replaced
        }
        if (c[k] > 0) {
            adj[c[k] * max_adj_size + adj_sizes[c[k]]] = a[k]; adj_sizes[c[k]]++; // m with max_adj_size replaced
            adj[c[k] * max_adj_size + adj_sizes[c[k]]] = b[k]; adj_sizes[c[k]]++; // m with max_adj_size replaced
        }
        else {
            adj[(-c[k] + n) * max_adj_size + adj_sizes[(-c[k] + n)]] = a[k]; adj_sizes[(-c[k] + n)]++; // m with max_adj_size replaced
            adj[(-c[k] + n) * max_adj_size + adj_sizes[(-c[k] + n)]] = b[k]; adj_sizes[(-c[k] + n)]++; // m with max_adj_size replaced
        }
    }
    std::cout<<"\r"<<log_time(); printf(" [INFO] BUILDING CIRCUIT(1/2): %d/%d ", m, m);
    printf("\n");
    //output ADJ:
    if (debug) {
        for (int k = 1; k < (n * 2 + 1); k++) {
            printf(" [INFO] DEBUG ADJ[%d]: ", k);
            for (j = 0; j < adj_sizes[k]; j++) {
                printf("%d ", adj[k * max_adj_size + j]);
            }
            printf("\n");
        }
    }
    //std::cout <<log_time(); printf(" [INFO] ADJANCES TABLE BUILT.\n");

    /// get negative associations: WE ASSUME MAX_ADJ_SIZE IS THE SAME FOR OPP LITERALS - CHECK THIS!!!!
    adj_opp_sizes = (int*)calloc((size_t)(2 * n + 1), sizeof(int));
    for (j = 1; j <= 2 * n; j++) adj_opp_sizes[j] = 0;

    adj_opp = (int*)calloc((size_t)max_adj_size * (2 * n + 1), sizeof(int));

    for (int i = 1; i <= 2 * n; i++) {
        if (i % 1000 == 0) {
		std::cout<<"\r"<<log_time();  printf(" [INFO] BUILDING CIRCUIT(2/2): %d/%d ", i, 2 * n);
        }
	int literal = ((i > n) ? (i - n) : (i + n));
        for (int k = 0; k < adj_sizes[literal]; k++) {
            int item2 = adj[literal * max_adj_size + k]; // m with max_adj_size replaced
            adj_opp[i * max_adj_size + adj_opp_sizes[i]] = ((item2 < 0) ? (n - item2) : item2); adj_opp_sizes[i]++;
            //printf("DEBUG: %d -> %d ",item2,((item2 < 0) ? (n - item2) : item2));
        }
        int* adj1 = (int*)calloc((size_t)max_adj_size, sizeof(int)); int adj1_pos = 0;
        int* adj2 = (int*)calloc((size_t)max_adj_size, sizeof(int)); int adj2_pos = 0;
        for (int j = 0; j < (adj_opp_sizes[i] + 1); j += 2)
        {
            adj1[adj1_pos] = adj_opp[i * max_adj_size + j]; adj1_pos++;
            adj2[adj2_pos] = adj_opp[i * max_adj_size + j + 1]; adj2_pos++;
        }
        for (int k = 0; k < adj_opp_sizes[i]; k++) adj_opp[i * max_adj_size + k] = 0; adj_opp_sizes[i] = 0; // clear

        for (int k = 0; k < (adj1_pos - 1); k++)
        {
            int val = adj1[k];
            int item = ((val > n) ? (val - n) : (val + n));
            int ajd1_found = 0; for (int x = 0; x < adj1_pos; x++) if (adj1[x] == item) ajd1_found = 1;
            if (ajd1_found == 1)
            {
                adj1[k] = adj2[k];
                adj2[k] = val;
            }
            val = adj2[k];
            item = ((val > n) ? (val - n) : (val + n));
            int ajd2_found = 0; for (int x = 0; x < adj2_pos; x++) if (adj2[x] == item) ajd2_found = 1;
            if (ajd2_found == 1)
            {
                adj2[k] = adj1[k];
                adj1[k] = val;
            }
        }
        /*
        printf("DEBUG %d: ",i);
        for (int q=0; q<adj1_pos; q++) {
            printf(" adj1[%d]=%d adj2[%d]=%d ",q,adj1[q],q,adj2[q]);
        }
        printf("\n");
        */

        for (int l = 0; l < (adj1_pos - 1); l++)
        {
            adj_opp[i * max_adj_size + adj_opp_sizes[i]] = adj1[l]; adj_opp_sizes[i]++;
            adj_opp[i * max_adj_size + adj_opp_sizes[i]] = adj2[l]; adj_opp_sizes[i]++;
        }
    }
    std::cout<<"\r"<<log_time();  printf(" [INFO] BUILDING CIRCUIT(2/2): %d/%d ", 2*n, 2 * n);
    printf("\n");

    //output:
    if (debug) {
        for (int k = 1; k < (n * 2 + 1); k++) {
            printf(" [INFO] DEBUG ADJ_OPP[%d]: ", k);
            for (j = 0; j < adj_opp_sizes[k]; j++) {
                printf("%d ", adj_opp[k * max_adj_size + j]);
            }
            printf("\n");
        }
    }
    //std::cout <<log_time(); printf(" [INFO] NEGATIVE ADJANCES TABLE BUILT.\n");
    //std::cout << "DEBUG: max_adj_size = " << max_adj_size << " table size = " << max_adj_size * (2 * n + 1) << std::endl;

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// DEVICE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// lambda: ------------------------------------------------------------------------------------------------
__device__  void lambda_init(job_struct& job) {
    for (int i = 0; i < job.lambda_pos; i++) {
        //job.lambda[i] = 0; //not required, _bin does this anyways
        job.lambda_bin[i] = false;
    }
    job.lambda_pos = 0;
    return;
}

__device__ __forceinline__ void set_lambda(const int _Xk, job_struct& job) {
    job.lambda[job.lambda_pos++] = _Xk;
    job.lambda_bin[_Xk] = true;
    return;
}

__device__ __forceinline__ void lambda_remove_last(job_struct& job) {
    job.lambda_bin[job.lambda[job.lambda_pos - 1]] = false;
    //job.lambda[job.lambda_pos - 1] = 0; //not required, _bin does this anyways
    job.lambda_pos--;
    return;
}

__device__ __forceinline__ void lambda_remove(const int _Xk, job_struct& job) {
    job.lambda_bin[_Xk] = false;
    /* sure about this?
    int lambda_set = 0;

    for (int i = 0; i < job.lambda_pos; i++) { // CHECK THIS FUNCTION
        if (job.lambda[i] != _Xk) {
            job.lambda[lambda_set++] = job.lambda[i];
        }
    }
    job.lambda_pos = lambda_set;
    */
    job.lambda_pos--;
    return;
}

__device__  void lambda_insertfirst(const int _Xk, job_struct& job) {
	for (int i = job.lambda_pos; i > 0; i--) {
        job.lambda[i] = job.lambda[i - 1];
    }
    job.lambda[0] = _Xk;
    job.lambda_pos++;
    job.lambda_bin[_Xk] = true;
    return;
}

__device__  job_struct lambda_update(job_struct& job) {
	for (int i = 0; i < job.n * 2 + 1; i++) job.lambda_bin[i] = false;
    for (int i = 0; i < job.lambda_pos; i++) job.lambda_bin[job.lambda[i]] = true;
    return;
}
__device__  __forceinline__ bool lambda_contains(const int _Xk, const bool* lambda_bin) {
    return lambda_bin[_Xk];
}

/* replace with restoreLambdaComplete - faster 
__device__  void RestoreLambda(const int previousSize, job_struct& job) { // make lambda the last previousSize items
	int diff = job.lambda_pos - previousSize; // lambda.Count - previousSize;
    if (diff > 0) {
    	//printf("----- RestoreLambda %d -------- \n",previousSize);
    	//printf("GPU DEBUG: "); for (int i=0; i<job.lambda_pos; i++) printf("%d ",job.lambda[i]); printf("\n");
        for (int i = 0; i < diff; i++) {
            //lambda.RemoveAt(0); // CHECK THIS
            //job.lambda_bin[0] = false;
            for (int j = 0; j < job.lambda_pos; j++) job.lambda[j] = job.lambda[j + 1];
            //dm job.lambda[job.lambda_pos - 1] = 0;
            job.lambda_pos--;
        }
        job.complexity_counter += diff;
        //printf("GPU ====> "); for (int i=0; i<job.lambda_pos; i++) printf("%d ",job.lambda[i]); printf("\n");
    }
    return;
}
*/

// replaces combo RestoreLambda:
__device__ void RestoreLambdaComplete(const int previousSize, job_struct& job){
	
	int diff = job.lambda_pos - previousSize;
	if (diff < 1) return;
	//printf("----- RestoreLambda %d -------- \n",previousSize);
	//printf("GPU DEBUG: "); for (int i=0; i<job.lambda_pos; i++) printf("%d ",job.lambda[i]); printf("\n");
	for (int i=0; i<job.lambda_pos; i++) {
		if (i<previousSize) {
			job.lambda[i] = job.lambda[i+diff];
			job.lambda_bin[job.lambda[i+diff]] = true;
		} else {
			job.lambda_bin[job.lambda[i+diff]] = false;
			job.lambda[i] = 0;
		}
	}
	job.lambda_pos-= diff;
	job.complexity_counter += diff;
	//printf("GPU ====> "); for (int i=0; i<job.lambda_pos; i++) printf("%d ",job.lambda[i]); printf("\n");
	return;
}


// new_units: -----------------------------------------------------------------------------------------------------------
__device__  void init_new_units(job_struct& job) {
    for (int i = 0; i < job.n * 2 + 1; i++) {
        job.new_units_bin[i] = false;
        //job.new_units[i] = 0; //dm speedup * 2
    }
    job.new_units_pos = 0;
    return;
}
__device__ __forceinline__ void set_new_units(const int _Xk, job_struct& job) {
    job.new_units[job.new_units_pos++] = _Xk;
    job.new_units_bin[_Xk] = true;
    return;
}

/*
__device__ __forceinline__ void new_units_remove_last(job_struct& job) {
    job.new_units_bin[job.new_units_pos - 1] = false;
    memcpy(&job.new_units[0], &job.new_units[1], sizeof(int) * (job.new_units_pos - 1));
    job.new_units_pos--; //left shift new_units by one (same speed)
    return;
}
*/

__device__  void new_units_remove(const int _Xk, job_struct& job) {
    job.new_units_bin[_Xk] = false;
    int units_set = 0;
    for (int i = 0; i < job.new_units_pos; i++) { // CHECK THIS FUNCTION
        if (job.new_units[i] != _Xk) {
            job.new_units[units_set++] = job.new_units[i];
        }
    }
    job.new_units_pos = units_set;
    return;
}
__device__ __forceinline__ bool new_units_contains(const int _Xk, const bool* new_units_bin) {
    return new_units_bin[_Xk];
}
__device__ __forceinline__ int Opposite(const int k, const int n) {
    return (k > n) ? (k - n) : (k + n);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
adj_opp memory optimization
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void FindUnits(int Xk, job_struct& job, const int* d_adj_opp, const int* d_adj_opp_sizes) {
    int val_b = 0;
    int val_c = 0;

    for (int i = 0; i < d_adj_opp_sizes[Xk]; i += 2) // adj_opp[Xk].Count; i += 2)
    {
        val_b = d_adj_opp[Xk * job.max_adj_size + i];
        val_c = d_adj_opp[Xk * job.max_adj_size + i + 1];
        job.complexity_counter += 2;
        if (lambda_contains(Opposite(val_b, job.n), job.lambda_bin) && !lambda_contains(val_c, job.lambda_bin) && !new_units_contains(val_c, job.new_units_bin))
        {
            set_new_units(val_c, job); //units.Add(val_c);
            job.complexity_counter += 4;
        }
        else if (lambda_contains(Opposite(val_c, job.n), job.lambda_bin) && !lambda_contains(val_b, job.lambda_bin) && !new_units_contains(val_b, job.new_units_bin))
        {
            set_new_units(val_b, job); //units.Add(val_b);
            job.complexity_counter += 4;
        }

    }
    return;
}

__device__ bool GetUnits(int Xk, job_struct& job, const int* d_adj_opp, const int* d_adj_opp_sizes) {

    //printf(" [INFO]    GetUnits()....\n");
    int cnt = job.lambda_pos;
    bool isSat = true;
    int unit = 0;
    int opp_unit = 0;

    init_new_units(job); //List<int> new_units = new List<int>();

    if (job.lambda_pos > 1) // (lambda.Count > 1)
    {
        FindUnits(Xk, job, d_adj_opp, d_adj_opp_sizes);
        job.complexity_counter++;

        //cannot exist more than n units
        for (int i = 1; i <= job.n && isSat && job.new_units_pos > 0; i++)
        {
            //pick the first unit in line
            unit = job.new_units[0];
            opp_unit = Opposite(unit, job.n);
            isSat = !lambda_contains(opp_unit, job.lambda_bin) && !new_units_contains(opp_unit, job.new_units_bin);

            job.complexity_counter += 4;
            if (isSat)
            {
                //and if there is no collision then insert it to lambda
                lambda_insertfirst(unit, job); // lambda.Insert(0, unit);
                new_units_remove(unit, job); //new_units.Remove(unit);
                //then use that unit to search for more
                FindUnits(unit, job, d_adj_opp, d_adj_opp_sizes);

                job.complexity_counter += 3;
            }
        }
    }

    //collision detected - restore lambda
    if (!isSat) {
    	RestoreLambdaComplete(cnt, job);
        //RestoreLambda(cnt, job); //job = RestoreLambda(cnt, job); 
        lambda_update(job);//job = lambda_update(job);
    }

    job.isSat = isSat;
    return isSat;
}

__device__ bool GetOppUnits(const int Xk, const bool pol, job_struct& job, const int* d_adj_opp, const int* d_adj_opp_sizes) {

	//printf(" [INFO] GetOppUnits...\n");
    bool isSat = true;
    int cnt = job.lambda_pos; // lambda.Count();
    for (int i = 0; i < d_adj_opp_sizes[Xk] && isSat; i += 2)
    {

        int val_b = pol ? d_adj_opp[Xk * job.max_adj_size + i] : d_adj_opp[Xk * job.max_adj_size + i + 1];
        int val_c = pol ? d_adj_opp[Xk * job.max_adj_size + i + 1] : d_adj_opp[Xk * job.max_adj_size + i];

        isSat = !(lambda_contains(Opposite(val_b,job.n), job.lambda_bin) && lambda_contains(Opposite(val_c,job.n),job.lambda_bin)) && ((lambda_contains(val_b, job.lambda_bin) || lambda_contains(val_c, job.lambda_bin) || GetUnits(val_b, job, d_adj_opp, d_adj_opp_sizes) || GetUnits(val_c, job, d_adj_opp, d_adj_opp_sizes)));

        //printf(" [INFO] GetOppUnits checkpoint 1 done.\n");

        job.complexity_counter += 8;
    }
    //printf(" [INFO] GetOppUnits checkpoint 2\n");
    //collision detected - restore lambda
    if (!isSat) {
    	RestoreLambdaComplete(cnt, job); //job = RestoreLambda(cnt, job); 
        lambda_update(job); //job = lambda_update(job);
    }
    //printf(" [INFO] GetOppUnits checkpoint 3\n");

    job.isSat = isSat;
    return isSat;
}

// faster than void: w/o: 1,918,747.875 kFLOPS with: 737,264.125 kFLOPS
// because this is the only function triggered from main loop
__device__ job_struct GetAllUnits(job_struct job, const int* d_adj_opp, const int* d_adj_opp_sizes) {

	int cnt = job.lambda_pos; // lambda.Count;
    bool isSat = true;

    for (int k = 0; k < cnt && isSat; k++)
    {
        job.complexity_counter++;
        isSat = GetUnits(job.lambda[k], job, d_adj_opp, d_adj_opp_sizes);
    }
    
    if (isSat)
    {
        for (int k = 0; k < cnt && isSat; k++)
        {
            job.complexity_counter++;
            isSat = GetOppUnits(job.lambda[k], job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
        }
        for (int k = 0; k < cnt && isSat; k++)
        {
            job.complexity_counter++;
            isSat = GetOppUnits(job.lambda[k], !job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
        }
    
        if (!isSat)
        {
            RestoreLambdaComplete(cnt, job); //job = RestoreLambda(cnt, job); 
            lambda_update(job); //job = lambda_update(job);
            isSat = true;
            switch (job.stage)
            {
            case 0:
                for (int k = 0; k < cnt && isSat; k++)
                {
                    job.complexity_counter++;
                    isSat = GetOppUnits(job.lambda[k], job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
                }
                break;
            case 1:
                for (int k = 0; k < cnt && isSat; k++)
                {
                    job.complexity_counter += 2;
                    isSat = GetOppUnits(job.lambda[k], job.polarity, job, d_adj_opp, d_adj_opp_sizes) || GetOppUnits(job.lambda[k], !job.polarity, job, d_adj_opp, d_adj_opp_sizes);
                }
                break;
            case 2:
                for (int k = 0; k < cnt && isSat; k++)
                {
                    job.complexity_counter++;
                    isSat = GetOppUnits(job.lambda[k], !job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
                }
                for (int k = 0; k < cnt && isSat; k++)
                {
                    job.complexity_counter++;
                    isSat = GetOppUnits(job.lambda[k], job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
                }
                break;

            default:
                for (int k = 0; k < cnt && isSat; k++)
                {
                    job.complexity_counter++;
                    isSat = GetOppUnits(job.lambda[k], !job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
                }
                if (!isSat)
                {
                    RestoreLambdaComplete(cnt, job); //job = RestoreLambda(cnt, job); 
                    lambda_update(job);//job = lambda_update(job);
                    isSat = true;
                    for (int k = 0; k < cnt && isSat; k++)
                    {
                        job.complexity_counter++;
                        isSat = GetOppUnits(job.lambda[k], job.polarity, job, d_adj_opp, d_adj_opp_sizes); 
                    }
                }
                break;
            }
        }
    
    }

    //collision detected - restore lambda
    if (!isSat) {
        RestoreLambdaComplete(cnt, job); //job = RestoreLambda(cnt, job); 
        lambda_update(job); //job = lambda_update(job);
    }

    job.isSat = isSat;
    return job;
    
}

__device__ bool IncrementHeader(job_struct& job)
{
    //printf(" [INFO] starting IncrementHeader...\n");
    bool isSat = true;

    job.polarity = !job.polarity;

    if (job.polarity) job.stage++;

    job.Xk = job.starting_Xk;
    set_lambda(job.Xk, job); //   lambda.Add(Xk);

    if (job.stage > 3)
    {
        job.stage = 0;
        job.polarity = true;
        lambda_init(job); //lambda.Clear();

        int header_set = 0;
        for (int i = 0; i < (job.n * 2 + 1); i++) {
            if (job.header[i]) header_set++;
        }
        if (header_set != 2 * job.n) isSat = false;

        //if not all literals have been used as headers

        if (isSat)
        {
            //pick the next unused literal
            for (int i = 1; i <= 2 * job.n && job.header[job.starting_Xk]; i++)
                if (job.header[job.starting_Xk]) job.starting_Xk = job.starting_Xk % (2 * job.n) + 1;
            job.header[job.starting_Xk] = true;
            job.Xk = job.starting_Xk;
            set_lambda(job.Xk, job); //lambda.Add(Xk);
        }
    }
    job.isSat = isSat;
    return isSat;
}

__device__ bool CheckBinarySolution(int* d_solved, const int* d_a, const int* d_b, const int* d_c, job_struct job, int* d_lambda_solution) {
    printf("[GPU] CHIP %d: LAMBDA REACHED MAXIMUM (starting=X%d stage=%d polarity=%d) Checking Binary Solution...\n", job.threadi, job.starting_Xk, job.stage, job.polarity);
    //condition 1
    bool isSat = job.lambda_pos == job.n;
    printf("[GPU] CHIP %d: CONDITION 1 OK (lambda = n).\n", job.threadi);

    //condition 2
    if (isSat) {
        for (int i = 0; i < job.lambda_pos; i++) if (lambda_contains(Opposite(job.lambda[i], job.n), job.lambda_bin)) isSat = false;
        if (isSat) printf("[GPU] CHIP %d: CONDITION 2 OK (no conflicting literals in lambda).\n", job.threadi);
    }

    if (isSat) {
        int satisfied_clauses = 0;
        for (int i = 0; i < job.m; i++) {
            int _a = (d_a[i] > 0) ? d_a[i] : abs(d_a[i] - job.n);
            int _b = (d_b[i] > 0) ? d_b[i] : abs(d_b[i] - job.n);
            int _c = (d_c[i] > 0) ? d_c[i] : abs(d_c[i] - job.n);
            if (lambda_contains(_a, job.lambda_bin) || lambda_contains(_b, job.lambda_bin) || lambda_contains(_c, job.lambda_bin)) {
                satisfied_clauses++;
            }
            else {
                printf("[GPU] CLAUSE %d [%d %d %d] -> [%d %d %d] NOT SATISFIED.\n", i, d_a[i], d_b[i], d_c[i], _a, _b, _c);
                isSat = false;
                break;
            }
        }
        if (satisfied_clauses == job.m) {
            printf("[GPU] CHIP %d: CONDITION 3 OK (all clauses SAT).\n", job.threadi);
            printf("[GPU] CHIP %d: PERFORMED %llu STEPS\n", job.threadi, job.complexity_counter);
            double ocompl = logf(job.complexity_counter) / logf(job.n);
            printf("[GPU] CHIP %d: COMPLEXITY (O)n^%f\n", job.threadi, ocompl);
#ifdef WIN32      
            printf("[GPU] CHIP %d: SOLUTION FOUND.\n", job.threadi);
#else
            printf("[GPU] CHIP %d: SOLUTION FOUND.\n", job.threadi);
#endif

            // if not already solved, let everyone know who did:
            if (d_solved[0] == 0) {
                atomicAdd(d_solved, (job.threadi+1)); // solved threadi => d_solved - need to be +1 because threadi = 0 would be no solution
                // move solution into d_lambda_solution
                for (int i = 0; i < job.n; i++) {
                    d_lambda_solution[i] = job.lambda[i];
                }
            }
        }
    }
    return isSat;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// GPU KERNELS
///////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void test_flops(uint64_cu operations) {
	__shared__ float x;
	x = 993212.123321;
	for (uint64_cu i=0; i<operations; i++) {
		x = x - 0.0000000001;
	}
	x = x + 1;
}

__global__
void init_dynex_mem(const int numchips, const int n, const int m, const int max_adj_size, job_struct * d_jobs) {

	/*d_jobs = new job_struct[numchips];
    d_a = new int[m];
    d_b = new int[m];
    d_c = new int[m];
    d_adj_opp = new int[max_adj_size * (2 * n + 1)];
    d_adj_opp_sizes = new int[(2 * n + 1)];
	*/
}

__global__
void init_dynex_jobs(const int numchips, const int n, const int m, const int max_adj_size, job_struct * d_jobs) {
    //printf("KERNEL init_dynex_jobs(%d)\n", numchips);
    //for (int chip = 0; chip < numchips; chip++) {
    //for (int chip = blockDim.x * blockIdx.x + threadIdx.x; chip < numchips; chip += gridDim.x * blockDim.x) {
    int chip = blockIdx.x * blockDim.x + threadIdx.x;
    if (chip < numchips) {
    	//d_jobs[chip].lambda = new int[n]; 
		//d_jobs[chip].lambda_bin = new bool[n * 2 + 1]; 
		//d_jobs[chip].header = new bool[n * 2 + 1]; 
		//d_jobs[chip].new_units_bin = new bool[2 * n + 1]; 
    	//d_jobs[chip].new_units = new int[2 * n + 1];
    	//d_jobs[chip].lambda = d_lambda_template;

        d_jobs[chip].threadi = -1;
        for (int i = 0; i < n; i++) d_jobs[chip].lambda[i] = 0;
        d_jobs[chip].lambda_pos = 0;
        for (int i = 0; i < (n * 2 + 1); i++) d_jobs[chip].lambda_bin[i] = false;
        d_jobs[chip].complexity_counter = (uint64_cu)0;
        d_jobs[chip].n = n;
        d_jobs[chip].m = m;
        d_jobs[chip].stage = 0;
        d_jobs[chip].polarity = false;
        d_jobs[chip].starting_Xk = -1;
        d_jobs[chip].Xk = -1;
        for (int i = 0; i < (n * 2 + 1); i++) d_jobs[chip].header[i] = false;
        d_jobs[chip].isSat = true;
        d_jobs[chip].new_units_pos = 0;
        for (int i = 0; i < (n * 2 + 1); i++) d_jobs[chip].new_units_bin[i] = false;
        d_jobs[chip].max_adj_size = max_adj_size;
        d_jobs[chip].flipped = true;
		
    	
    }
    //printf("KERNEL prepared d_jobs for %d chips\n", numchips);
    
}

__global__
void init_dynex_abc(const int m, int* a_tmp, int* b_tmp, int* c_tmp, int * d_a, int * d_b, int * d_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        d_a[i] = a_tmp[i];
        d_b[i] = b_tmp[i];
        d_c[i] = c_tmp[i];
    }
}

__global__
void init_dynex(const int numchips, const int n, const int m, const int max_adj_size, int *adj_opp_tmp, int *adj_opp_sizes_tmp, int * d_adj_opp, int * d_adj_opp_sizes) {
    // adj_opp:
    for (int i = 0; i < (max_adj_size * (2 * n + 1)); i++) d_adj_opp[i] = adj_opp_tmp[i];
    // adj_opp_sizes:
    for (int i = 0; i < (2 * n + 1); i++) d_adj_opp_sizes[i] = adj_opp_sizes_tmp[i];
    return;
}

// get state for one chip:
__global__
void get_state(const int i, job_struct * d_job, int * d_job_lambda, bool * d_job_lambda_bin, bool * d_job_header, int * d_job_new_units, bool * d_job_new_units_bin, job_struct * d_jobs) {
    d_job[0] = d_jobs[i];
    for (int j = 0; j < d_jobs[i].n; j++) d_job_lambda[j] = d_jobs[i].lambda[j];//copy lambda
    for (int j = 0; j < 2 * d_jobs[i].n + 1; j++) {
        d_job_lambda_bin[j] = d_jobs[i].lambda_bin[j];//copy lambda_bin
        d_job_header[j] = d_jobs[i].header[j];//copy header
        d_job_new_units[j] = d_jobs[i].new_units[j];//copy new_units
        d_job_new_units[j] = d_jobs[i].new_units_bin[j];//copy new_units_bin
    }
}

// set state for one chip:
__global__
void set_state(const int i, job_struct* d_job, int* d_job_lambda, bool* d_job_lambda_bin, bool* d_job_header, int* d_job_new_units, bool* d_job_new_units_bin, job_struct * d_jobs) {
   
    //d_jobs[i] = d_job[0];
    d_jobs[i].threadi = d_job[0].threadi;
    d_jobs[i].lambda_pos = d_job[0].lambda_pos;
    d_jobs[i].complexity_counter = d_job[0].complexity_counter;
    d_jobs[i].n = d_job[0].n;
    d_jobs[i].m = d_job[0].m;
    d_jobs[i].stage = d_job[0].stage;
    d_jobs[i].polarity = d_job[0].polarity;
    d_jobs[i].new_units_pos = d_job[0].new_units_pos;
    d_jobs[i].isSat = d_job[0].isSat;
    d_jobs[i].Xk = d_job[0].Xk;
    d_jobs[i].starting_Xk = d_job[0].starting_Xk;
    d_jobs[i].max_adj_size = d_job[0].max_adj_size;
    d_jobs[i].flipped = d_job[0].flipped;

    for (int j = 0; j < d_jobs[i].n; j++) d_jobs[i].lambda[j] = d_job_lambda[j];//copy lambda
    for (int j = 0; j < 2 * d_jobs[i].n + 1; j++) {
        d_jobs[i].lambda_bin[j] = d_job_lambda_bin[j];//copy lambda_bin
        d_jobs[i].header[j] = d_job_header[j];//copy header
        d_jobs[i].new_units[j] = d_job_new_units[j];//copy new_units
        d_jobs[i].new_units_bin[j] = d_job_new_units_bin[j];//copy new_units_bin
    }
}

__global__
void run_DynexChipUpperBound(const int dev, const int n, const int m, const int max_adj_size, int* d_solved, int* d_lambda_solution, uint64_cu * d_total_steps, const bool init, const uint64_cu maxsteps, const int jobs_required_from, const int jobs_required_to, job_struct * d_jobs, const int * d_a, const int * d_b, const int *d_c, const int * d_adj_opp, const int * d_adj_opp_sizes) {

    int threadi = blockIdx.x * blockDim.x + threadIdx.x;

    const int loop_to = jobs_required_to - jobs_required_from;
    //printf(" [INFO] [GPU %d]: jobs_required_from = %d , jobs_required_to = %d\n", dev, jobs_required_from, jobs_required_to);
    
    // loop only in range jobs_required_from - jobs_required_to
    while (threadi >= 0 && threadi <= loop_to) { // VERIFY IF <= or < IS CORRECT TODO
        
        // job definition init (first run)? otherwise we continue
        if (init) {
            // convert threadi to starting_Xk, stage and polarity:
            int threadi_shifted = threadi + jobs_required_from;
            int starting_Xk = int(threadi_shifted / 8) + 1;
            int stage = threadi_shifted % 4;
            bool polarity = threadi_shifted % 8 < 4 ? true : false;
            assert(starting_Xk < (n * 2) + 1);
            assert(stage < 4);
            assert(polarity == true || polarity == false);
            //printf(" [INFO] [GPU %d] THREAD %d:  starting_Xk=%d stage=%d polarity=%d\n",dev,threadi,starting_Xk,stage,polarity);

            // init job details: 
            d_jobs[threadi].threadi = threadi; 
            d_jobs[threadi].lambda_pos = 0;
            d_jobs[threadi].complexity_counter = 0;
            d_jobs[threadi].n = n;
            d_jobs[threadi].m = m;
            d_jobs[threadi].stage = stage;
            d_jobs[threadi].polarity = polarity;
            d_jobs[threadi].starting_Xk = starting_Xk;
            d_jobs[threadi].Xk = starting_Xk;
            d_jobs[threadi].isSat = true;
            d_jobs[threadi].header[starting_Xk] = true;
            d_jobs[threadi].new_units_pos = 0;
            d_jobs[threadi].max_adj_size = max_adj_size;
            d_jobs[threadi].flipped = true;
            //set lambda:
            set_lambda(d_jobs[threadi].Xk, d_jobs[threadi]);

        }

        // solving sequence --------------------------------------------------------------------------------
        bool loopFurther = true;
        bool isSat = true; 

        // loop until next complexity:    
        uint64_cu loop_until_steps = d_jobs[threadi].complexity_counter + maxsteps;
        
        // main loop: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        uint64_cu complexity_plus = 0;
        
        while (d_solved[0] == 0 && d_jobs[threadi].complexity_counter < loop_until_steps && loopFurther) {
    
            for (int i = 1; i <= 2 * n; i++) {
                if (lambda_contains(d_jobs[threadi].Xk, d_jobs[threadi].lambda_bin) || lambda_contains(Opposite(d_jobs[threadi].Xk, n), d_jobs[threadi].lambda_bin)) {
                    d_jobs[threadi].Xk = d_jobs[threadi].Xk % (2 * n) + 1;
                    d_jobs[threadi].complexity_counter += 4;
                }
                else {
                    break;
                }
            }
            
            set_lambda(d_jobs[threadi].Xk, d_jobs[threadi]);
            d_jobs[threadi].complexity_counter++;

            // calling the main execution block:
            d_jobs[threadi] = GetAllUnits(d_jobs[threadi], d_adj_opp, d_adj_opp_sizes); 
            isSat = d_jobs[threadi].isSat;

            if (!isSat || (d_jobs[threadi].lambda_pos == n && !CheckBinarySolution(d_solved, d_a, d_b, d_c, d_jobs[threadi], d_lambda_solution)))
            {
                lambda_remove(d_jobs[threadi].Xk, d_jobs[threadi]); //lambda.Remove(Xk);
                d_jobs[threadi].Xk = Opposite(d_jobs[threadi].Xk, n);
                d_jobs[threadi].complexity_counter += 2;
                d_jobs[threadi].flipped = !d_jobs[threadi].flipped;
                if (d_jobs[threadi].flipped)
                {
                    if (d_jobs[threadi].lambda_pos > 0)
                        lambda_remove_last(d_jobs[threadi]); //lambda.Remove(lambda.Last());

                    d_jobs[threadi].complexity_counter++;
                }
            }
            else
                d_jobs[threadi].flipped = true;

            loopFurther = (d_jobs[threadi].lambda_pos != 0 || IncrementHeader(d_jobs[threadi])) && d_jobs[threadi].lambda_pos < n;

        }
        // main loop finished... +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        // increase number of steps:
        atomicAdd(d_total_steps, d_jobs[threadi].complexity_counter);
        // did someone find a solution? then we stop
        if (d_solved[0] != 0) {
            return;
        }
        threadi += blockDim.x * gridDim.x; // try next threadi
    }
    
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Query devices
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int query_devices() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf(" [INFO] %d GPU(s) FOUND:\n", nDevices);
    for (int i = 0; i < nDevices; i++) {
        std::cout << "[INFO] CUDA DEVICE " << i << " -------------------------------------------------------" << std::endl;
        cudaSetDevice(i);
        int runtimeVersion;
        cudaRuntimeGetVersion(&runtimeVersion);
        int driverVersion;
        cudaDriverGetVersion(&driverVersion);
        printf(" [INFO] Runtime Version:               %d\n",runtimeVersion);
        printf(" [INFO] Driver Version:                %d\n",driverVersion);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf(" [INFO] SM Major revision number:      %d\n", devProp.major);
        printf(" [INFO] SM Minor revision number:      %d\n", devProp.minor);
        printf(" [INFO] Name:                          %s\n", devProp.name);
        printf(" [INFO] Total global memory:           %zu\n", devProp.totalGlobalMem);
        printf(" [INFO] Maximum 3D CUDA array:         %u\n", devProp.maxTexture3D[3]);
        printf(" [INFO] Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
        printf(" [INFO] Total registers per block:     %d\n", devProp.regsPerBlock);
        printf(" [INFO] Warp size:                     %d\n", devProp.warpSize);
        printf(" [INFO] Maximum memory pitch:          %zu\n", devProp.memPitch);
        printf(" [INFO] Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i)
            printf(" [INFO] Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf(" [INFO] Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf(" [INFO] Clock rate:                    %d\n", devProp.clockRate);
        printf(" [INFO] Total constant memory:         %zu\n", devProp.totalConstMem);
        printf(" [INFO] Texture alignment:             %zu\n", devProp.textureAlignment);
        printf(" [INFO] Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
        printf(" [INFO] Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
        printf(" [INFO] Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        if (devProp.major < 2)
        {
            printf("We require SM >= 2.0 to support Texture Arrays.  Test will be waived... \n");
            cudaDeviceReset();
            return false;
        }
    }
    return nDevices;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// write and load state (h_job,d_job) to disc
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// write state:
void write_states(int dev, int _numchips, int _CHIP_FROM, int _CHIP_TO, uint64_cu _h_total_steps) {
    //printf("write_states: writing gpu %d: %d jobs, chips %d-%d, %llu total steps\n", dev, _numchips, _CHIP_FROM, _CHIP_TO, _h_total_steps);
    
    DISC_OPERATION = true;
    std::cout << log_time() << TEXT_SILVER << " [GPU " << dev << "] SAVING STATE TO DISK - DO NOT INTERRUPT!..." << TEXT_DEFAULT << std::endl;

    // prepare binary file:
    ofstream outfile;
    std::string FN; FN = "GPU_" + std::to_string(dev) + ".bin";
    outfile.open(FN.c_str(), std::ios::trunc | std::ios::binary);
    outfile.write(MALLOB_NETWORK_ID.c_str(), 64);
    int filenamesize = JOB_FILENAME.size();
    outfile.write((char*)&filenamesize, sizeof(filenamesize));
    outfile.write(JOB_FILENAME.c_str(), filenamesize);
    outfile.write((char*)&_numchips, sizeof(_numchips));
    outfile.write((char*)&_CHIP_FROM, sizeof(_CHIP_FROM));
    outfile.write((char*)&_CHIP_TO, sizeof(_CHIP_TO));
    outfile.write((char*)&_h_total_steps, sizeof(_h_total_steps));

    // host data:
    job_struct* h_job           = new job_struct[1];
    int*  h_job_lambda          = new int[n];
    bool* h_job_lambda_bin      = new bool[n * 2 + 1];
    bool* h_job_header          = new bool[n * 2 + 1];
    int* h_job_new_units        = new int[n * 2 + 1];
    bool* h_job_new_units_bin   = new bool[n * 2 + 1];
    // device data:
    gpuErrchk(cudaSetDevice(dev));
    job_struct  * d_job;        gpuErrchk(cudaMalloc((void**)&d_job, sizeof(job_struct)));
    int* d_job_lambda;          gpuErrchk(cudaMalloc((void**)&d_job_lambda, n* sizeof(int))); 
    bool* d_job_lambda_bin;     gpuErrchk(cudaMalloc((void**)&d_job_lambda_bin, (n * 2 + 1) * sizeof(bool)));
    bool* d_job_header;         gpuErrchk(cudaMalloc((void**)&d_job_header, (n * 2 + 1) * sizeof(bool)));
    int* d_job_new_units;       gpuErrchk(cudaMalloc((void**)&d_job_new_units, (n * 2 + 1) * sizeof(int)));
    bool* d_job_new_units_bin;  gpuErrchk(cudaMalloc((void**)&d_job_new_units_bin, (n * 2 + 1) * sizeof(bool)));

    // loop through all chips:
    for (int i = 0; i < _numchips; i++) {
        std::cout << "\r" << log_time() << " [GPU "<<dev<<"] WRITING CHIP #" << i+1 << "/" << _numchips << " ";
        

        get_state << < 1, 1 >> > (i, d_job, d_job_lambda, d_job_lambda_bin, d_job_header, d_job_new_units, d_job_new_units_bin, d_jobs[dev]);
        gpuErrchk(cudaDeviceSynchronize());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
            return;
        }
        
        gpuErrchk(cudaMemcpy(h_job, d_job, sizeof(job_struct), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_job_lambda, d_job_lambda, n * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_job_lambda_bin, d_job_lambda_bin, (n * 2 + 1) * sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_job_header, d_job_header, (n * 2 + 1) * sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_job_new_units, d_job_new_units, (n * 2 + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_job_new_units_bin, d_job_new_units_bin, (n * 2 + 1) * sizeof(bool), cudaMemcpyDeviceToHost));

        // first line data:
        outfile.write((char*)&h_job[0].threadi, sizeof(h_job[0].threadi));
        outfile.write((char*)&h_job[0].lambda_pos, sizeof(h_job[0].lambda_pos));
        outfile.write((char*)&h_job[0].complexity_counter, sizeof(h_job[0].complexity_counter));
        outfile.write((char*)&h_job[0].n, sizeof(h_job[0].n));
        outfile.write((char*)&h_job[0].m, sizeof(h_job[0].m));
        outfile.write((char*)&h_job[0].stage, sizeof(h_job[0].stage));
        outfile.write((char*)&h_job[0].polarity, sizeof(h_job[0].polarity));
        outfile.write((char*)&h_job[0].new_units_pos, sizeof(h_job[0].new_units_pos));
        outfile.write((char*)&h_job[0].isSat, sizeof(h_job[0].isSat));
        outfile.write((char*)&h_job[0].Xk, sizeof(h_job[0].Xk));
        outfile.write((char*)&h_job[0].starting_Xk, sizeof(h_job[0].starting_Xk));
        outfile.write((char*)&h_job[0].max_adj_size, sizeof(h_job[0].max_adj_size));
        outfile.write((char*)&h_job[0].flipped, sizeof(h_job[0].flipped));

        // arrays:
        for (int j = 0; j < n; j++) outfile.write((char*)&h_job_lambda[j], sizeof(h_job_lambda[j]));
        for (int j = 0; j < (2 * n + 1); j++) outfile.write((char*)&h_job_lambda_bin[j], sizeof(h_job_lambda_bin[j]));
        for (int j = 0; j < (2 * n + 1); j++) outfile.write((char*)&h_job_header[j], sizeof(h_job_header[j]));
        for (int j = 0; j < (2 * n + 1); j++) outfile.write((char*)&h_job_new_units[j], sizeof(h_job_new_units[j]));
        for (int j = 0; j < (2 * n + 1); j++) outfile.write((char*)&h_job_new_units_bin[j], sizeof(h_job_new_units_bin[j]));
    }
    std::cout << "DONE." << std::endl;
    outfile.close();

    cudaFree(d_job);
    cudaFree(d_job_lambda);
    cudaFree(d_job_lambda_bin);
    cudaFree(d_job_header);
    cudaFree(d_job_new_units);
    cudaFree(d_job_new_units_bin);
    
    DISC_OPERATION = false;
    
}
// read state:
void read_states(int dev) {

    DISC_OPERATION = true;
	char _MALLOB_NETWORK_ID[64];
	char _JOB_FILENAME[1024];
	
	ifstream infile;
    std::string FN; FN = "GPU_" + std::to_string(dev) + ".bin";
	infile.open(FN.c_str(), std::ios::in | std::ios::binary);
	infile.read((char*)&_MALLOB_NETWORK_ID, 64);
	int filenamesize; 
	infile.read((char*)&filenamesize, sizeof(filenamesize));
	infile.read((char*)&_JOB_FILENAME, filenamesize);
	infile.read((char*)&num_jobs[dev], sizeof(num_jobs[dev]));
	infile.read((char*)&CHIP_FROM[dev], sizeof(CHIP_FROM[dev]));
	infile.read((char*)&CHIP_TO[dev], sizeof(CHIP_TO[dev]));
	infile.read((char*)&h_total_steps[dev], sizeof(h_total_steps[dev]));

    std::cout << log_time() << " [GPU " << dev << "] num_jobs = " << num_jobs[dev] << " CHIP_FROM = " << CHIP_FROM[dev] << " CHIP_TO = " << CHIP_TO[dev] << " h_total_steps = " << h_total_steps[dev] << std::endl;

    /// INIT MEMORY WITH KERNEL: ------------------------------------------------------------------------------------------
    gpuErrchk(cudaSetDevice(dev));

    std::cout << log_time() << " [GPU " << dev << "] ALLOCATING MEMORY... ";
	// new allocate from host:
    // job template:
    job_struct job_template;
    job_template.threadi = -1;
    job_template.lambda = new int[n]; for (int i = 0; i < n; i++) job_template.lambda[i] = 0;
    job_template.lambda_pos = 0;
    job_template.lambda_bin = new bool[n * 2 + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.lambda_bin[i] = false;
    job_template.complexity_counter = 0;
    job_template.n = n;
    job_template.m = m;
    job_template.stage = 0;
    job_template.polarity = false;
    job_template.starting_Xk = -1;
    job_template.Xk = -1;
    job_template.header = new bool[n * 2 + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.header[i] = false;
    job_template.isSat = true;
    job_template.new_units = new int[2 * n + 1];
    job_template.new_units_pos = 0;
    job_template.new_units_bin = new bool[2 * n + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.new_units_bin[i] = false;
    job_template.max_adj_size = max_adj_size;
    job_template.flipped = true;
    job_struct d_job_template;
    // create h_jobs and copy to d_jobs:
	int jobs_bytes = num_jobs[dev] * sizeof(job_template);
	h_jobs[dev] = (job_struct*)calloc((size_t) jobs_bytes, sizeof(size_t));
	for (int i = 0; i < num_jobs[dev]; i++) h_jobs[dev][i] = job_template;
	//copy jobs over to GPU (including sub arrays):
    uint64_cu mem_reserved = 0;
    for (int i = 0; i < num_jobs[dev]; i++) {
		gpuErrchk(cudaMalloc(&(h_jobs[dev][i].lambda), (n) * sizeof(int))); 
		gpuErrchk(cudaMalloc(&(h_jobs[dev][i].lambda_bin), (2 * n + 1) * sizeof(bool)));
		gpuErrchk(cudaMalloc(&(h_jobs[dev][i].header), (2 * n + 1) * sizeof(bool)));
		gpuErrchk(cudaMalloc(&(h_jobs[dev][i].new_units), (2 * n + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&(h_jobs[dev][i].new_units_bin), (2 * n + 1) * sizeof(bool)));
		mem_reserved += (2 * n + 1) * sizeof(bool) * 3 + n * sizeof(int) + (2 * n + 1) * sizeof(int);
    }
    std::cout << " DONE." << std::endl;
    std::cout << log_time() << " [GPU " << dev << "] ALLOCATED " << mem_reserved << " BYTES.";
    std::cout << std::endl;
    gpuErrchk(cudaMalloc((void**)&d_jobs[dev], jobs_bytes)); //reserve memory for all jobs
    gpuErrchk(cudaMemcpy(d_jobs[dev], h_jobs[dev] ,jobs_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&d_a[dev], m * sizeof(int))); // <== works
    gpuErrchk(cudaMalloc((void**)&d_b[dev], m * sizeof(int))); // <== works
    gpuErrchk(cudaMalloc((void**)&d_c[dev], m * sizeof(int))); // <== works
    gpuErrchk(cudaMalloc((void**)&d_adj_opp[dev], max_adj_size * (2*n+1) * sizeof(int))); // <== works
    gpuErrchk(cudaMalloc((void**)&d_adj_opp_sizes[dev], (2*n+1) * sizeof(int))); // <== works

    // d_jobs:
    std::cout << log_time() << " [GPU " << dev << "] INITIALIZING CHIPS... ";
    init_dynex_jobs << <  num_jobs[dev], 1 >> > (num_jobs[dev], n, m, max_adj_size, d_jobs[dev]);
    gpuErrchk(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
        return;
    }
    std::cout << "DONE" << std::endl;

    // d_a, d_b, d_c:
    std::cout << log_time() << " [GPU " << dev << "] COPYING PROBLEM... ";
    int* d_a_tmp;
    int* d_b_tmp;
    int* d_c_tmp;
    gpuErrchk(cudaMalloc((void**)&d_a_tmp, m * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_b_tmp, m * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_c_tmp, m * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_a_tmp, a, m * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_tmp, b, m * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c_tmp, c, m * sizeof(int), cudaMemcpyHostToDevice));
    init_dynex_abc << <  m, 1 >> > (m, d_a_tmp, d_b_tmp, d_c_tmp, d_a[dev], d_b[dev], d_c[dev]);
    gpuErrchk(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
        return;
    }
    std::cout << "DONE" << std::endl;

    // d_adj_opp, d_adj_opp_sizes:
    std::cout << log_time() << " [GPU " << dev << "] COPYING CHIP TABLES... ";
    int* d_adj_opp_tmp;
    gpuErrchk(cudaMalloc((void**)&d_adj_opp_tmp, max_adj_size * (2 * n + 1) * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_adj_opp_tmp, adj_opp, max_adj_size * (2 * n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    int* d_adj_opp_sizes_tmp;
    gpuErrchk(cudaMalloc((void**)&d_adj_opp_sizes_tmp, (2 * n + 1) * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_adj_opp_sizes_tmp, adj_opp_sizes, (2 * n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    init_dynex << < 1, 1 >> > (num_jobs[dev], n, m, max_adj_size, d_adj_opp_tmp, d_adj_opp_sizes_tmp, d_adj_opp[dev], d_adj_opp_sizes[dev]); // PREPARE d_jobs on GPU, d_a, d_b, d_c
    gpuErrchk(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
        return;
    }
    std::cout << "DONE" << std::endl;
    std::cout << log_time() << " [GPU " << dev << "] INITITALIZED." << std::endl;

    // now read the data:
    // host data:
    job_struct* h_job = new job_struct[1];
    int* h_job_lambda = new int[n];
    bool* h_job_lambda_bin = new bool[n * 2 + 1];
    bool* h_job_header = new bool[n * 2 + 1];
    int* h_job_new_units = new int[n * 2 + 1];
    bool* h_job_new_units_bin = new bool[n * 2 + 1];
    // device data:
    job_struct* d_job;          gpuErrchk(cudaMalloc((void**)&d_job, sizeof(job_struct)));
    int* d_job_lambda;          gpuErrchk(cudaMalloc((void**)&d_job_lambda, n * sizeof(int)));
    bool* d_job_lambda_bin;     gpuErrchk(cudaMalloc((void**)&d_job_lambda_bin, (n * 2 + 1) * sizeof(bool)));
    bool* d_job_header;         gpuErrchk(cudaMalloc((void**)&d_job_header, (n * 2 + 1) * sizeof(bool)));
    int* d_job_new_units;       gpuErrchk(cudaMalloc((void**)&d_job_new_units, (n * 2 + 1) * sizeof(int)));
    bool* d_job_new_units_bin;  gpuErrchk(cudaMalloc((void**)&d_job_new_units_bin, (n * 2 + 1) * sizeof(bool)));

    // read every chip:
    for (int i = 0; i < num_jobs[dev]; i++) {
        std::cout << "\r" << log_time() << " [GPU " << dev << "] READING CHIP #" << i+1 << "/" << num_jobs[dev] << " ";
        
        // first line data:
        infile.read((char*)&h_job[0].threadi, sizeof(h_job[0].threadi));
        infile.read((char*)&h_job[0].lambda_pos, sizeof(h_job[0].lambda_pos));
        infile.read((char*)&h_job[0].complexity_counter, sizeof(h_job[0].complexity_counter));
        infile.read((char*)&h_job[0].n, sizeof(h_job[0].n));
        infile.read((char*)&h_job[0].m, sizeof(h_job[0].m));
        infile.read((char*)&h_job[0].stage, sizeof(h_job[0].stage));
        infile.read((char*)&h_job[0].polarity, sizeof(h_job[0].polarity));
        infile.read((char*)&h_job[0].new_units_pos, sizeof(h_job[0].new_units_pos));
        infile.read((char*)&h_job[0].isSat, sizeof(h_job[0].isSat));
        infile.read((char*)&h_job[0].Xk, sizeof(h_job[0].Xk));
        infile.read((char*)&h_job[0].starting_Xk, sizeof(h_job[0].starting_Xk));
        infile.read((char*)&h_job[0].max_adj_size, sizeof(h_job[0].max_adj_size));
        infile.read((char*)&h_job[0].flipped, sizeof(h_job[0].flipped));

        // arrays:
        for (int j = 0; j < n; j++) infile.read((char*)&h_job_lambda[j], sizeof(h_job_lambda[j]));
        for (int j = 0; j < (2 * n + 1); j++) infile.read((char*)&h_job_lambda_bin[j], sizeof(h_job_lambda_bin[j]));
        for (int j = 0; j < (2 * n + 1); j++) infile.read((char*)&h_job_header[j], sizeof(h_job_header[j]));
        for (int j = 0; j < (2 * n + 1); j++) infile.read((char*)&h_job_new_units[j], sizeof(h_job_new_units[j]));
        for (int j = 0; j < (2 * n + 1); j++) infile.read((char*)&h_job_new_units_bin[j], sizeof(h_job_new_units_bin[j]));

        gpuErrchk(cudaMemcpy(d_job, h_job, sizeof(job_struct), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_job_lambda, h_job_lambda, n * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_job_lambda_bin, h_job_lambda_bin, (n * 2 + 1) * sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_job_header, h_job_header, (n * 2 + 1) * sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_job_new_units, h_job_new_units, (n * 2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_job_new_units_bin, h_job_new_units_bin, (n * 2 + 1) * sizeof(bool), cudaMemcpyHostToDevice));
        
        set_state << < 1, 1 >> > (i, d_job, d_job_lambda, d_job_lambda_bin, d_job_header, d_job_new_units, d_job_new_units_bin, d_jobs[dev]);
        gpuErrchk(cudaDeviceSynchronize());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
            return;
        }
       
    }
    std::cout << "DONE." << std::endl;
    
    infile.close();
	
    DISC_OPERATION = false;

    cudaFree(d_job);
    cudaFree(d_job_lambda);
    cudaFree(d_job_lambda_bin);
    cudaFree(d_job_header);
    cudaFree(d_job_new_units);
    cudaFree(d_job_new_units_bin);
	
	return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// start dynexsolve
///////////////////////////////////////////////////////////////////////////////////////////////////////////
bool run_dynexsolve(int start_from_job, int maximum_jobs, int steps_per_batch, int device_id, std::atomic<bool>& dynex_quit_flag, int work_in_progress) {
    
    // parallel runs required:
    // starting literal: n * 2 (positive and negative)
    // stage: 0,1,2,3
    // polarity: true, false
    
    // Dynex Service start:
    dynexservice.leffom = 0;
    if (!dynexservice.start(1, DAEMON_HOST, DAEMON_PORT, MINING_ADDRESS, 0, stratum, STRATUM_URL, STRATUM_PORT, STRATUM_PAYMENT_ID, STRATUM_PASSWORD)){ 
    	std::cout << log_time() << " [ERROR] CANNOT START DYNEX SERVICE." << std::endl;
    	return false;
    }

    // max upper bound:
    uint64_cu max_com = (uint64_cu)pow(n, 5); // std::numeric_limits<uint64_t>::max(); // CANNOT BE LARGER THAN uint_64 max
    std::cout << log_time() << " [INFO] UPPER BOUND COMPLEXITY: " << n * 2 * 8 << " PARALLEL DYNEX CHIPS, MAX O(n^5)=" << max_com << " STEPS" << std::endl;
    
    // GPU memories:
    // find maximum heap size which can be used for each GPU -> max_heap_size[nDevices]
    size_t* max_heap_size = (size_t*)calloc((size_t)nDevices, sizeof(size_t));
    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		size_t free, total; //tmp vars
		cudaMemGetInfo(&free, &total);
		size_t malloc_limit = free;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit);
		std::cout << log_time() << " [GPU " << device_id << "] SET cudaDeviceSetLimit: " << malloc_limit << " BYTES" << std::endl;
		cudaDeviceGetLimit(&max_heap_size[dev], cudaLimitMallocHeapSize);
		std::cout << log_time() << " [GPU " << device_id << "] GET cudaDeviceGetLimit: " << max_heap_size[dev] << " BYTES" << std::endl;        
        }
    }
    
    // init h_solved (single & multi):
    int* h_solved = (int*)calloc((size_t) 1, sizeof(int));
    h_solved[0] = 0;
    
    int* d_solved[MAX_NUM_GPUS]; // multi-gpu: threadi of who solved, otherwise 0
    uint64_cu* d_total_steps[MAX_NUM_GPUS]; // multi-gpu: total steps of each GPU

    int* d_lambda_solution[MAX_NUM_GPUS]{}; // contains lambda with solution if found
    for (int i = 0; i < nDevices; i++) d_lambda_solution[i] = new int(n);

    // move core (copyable) data to device (single & multi):
    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		gpuErrchk(cudaMalloc((void**)&d_solved[dev], 1 * sizeof(int))); // <== works
		gpuErrchk(cudaMemcpy(d_solved[dev], h_solved, sizeof(int), cudaMemcpyHostToDevice)); 
		gpuErrchk(cudaMalloc((void**)&d_total_steps[dev], sizeof(uint64_cu))); // <== works
		gpuErrchk(cudaMemcpy(d_total_steps[dev], h_total_steps, sizeof(uint64_cu), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc((void**)&d_lambda_solution[dev], n * sizeof(int))); // <== works
		std::cout << log_time() << " [GPU " << device_id <<"] CORE DATA COPIED TO GPU " << device_id << std::endl;
        }
    }

    // init states for GPU - LOOP trough devices
    
    int num_jobs_all = 0;
    uint64_cu* h_total_steps_dev = new uint64_cu[nDevices]; 
    uint64_cu h_total_steps_all = 0;
    
    // work in progress? load state from disc to GPU ----------------------------------------------------------------------
    if (work_in_progress) {
        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev; 
            // only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		    std::cout << log_time() << TEXT_SILVER << " [GPU " << device_id << "] CONTINUING WORK IN PROGRESS..." << TEXT_DEFAULT << std::endl;
		    read_states(device_id);
		    std::cout << log_time() << TEXT_SILVER << " [GPU " << device_id << "] CONTINUING FROM TOTAL STEPS = " << h_total_steps[dev] << TEXT_DEFAULT << std::endl;
		    // update data:
		    num_jobs_all += num_jobs[dev];
		    gpuErrchk(cudaSetDevice(device_id));
		    gpuErrchk(cudaMemcpy(d_total_steps[dev], h_total_steps, sizeof(uint64_cu), cudaMemcpyHostToDevice));
		    h_total_steps_all += h_total_steps[0];
		    h_total_steps_dev[dev] = h_total_steps[0];
            }
        }
        // work in progress initiated -------------------------------------------------------------------------------------------
    }
    else {
        // new work, initiate state and load to GPU -----------------------------------------------------------------------------
        // base:
        uint64_cu mem_req = 0; //1024 * 1024 * 1024; // OS
        mem_req += m * sizeof(int) * 3; // d_a, d_b, d_c
        mem_req += max_adj_size * (2 * n + 1) * sizeof(int); //d_adj_opp;
        mem_req += (2 * n + 1) * sizeof(int); //d_adj_opp_sizes;
        mem_req += n * sizeof(int); //d_lambda_solution
        mem_req += sizeof(int) * nDevices; //d_solved
        mem_req += sizeof(uint64_cu); //d_total_steps;
        
        // per job:
        uint64_cu mem_job = 0;
        mem_job += n * sizeof(int); //lambda
        mem_job += sizeof(int); //lambda_pos
        mem_job += (n * 2 + 1) * sizeof(bool); //lambda_bin
        mem_job += sizeof(uint64_cu); //complexity_counter
        mem_job += sizeof(int) * 6; //n,m,stage,starting_Xk, Xk, max_adj_size
        mem_job += sizeof(bool) * 3; //polariy, issat, flipped
        mem_job += (n * 2 + 1) * sizeof(bool); //header
        mem_job += (2 * n + 1) * sizeof(int); //units
        mem_job += sizeof(int); //units_pos
        mem_job += (2 * n + 1) * sizeof(bool); //units_bin
        
        mem_job = abs(mem_job * 1.50);
        
        // fitting jobs:
        int jobs_possible_all = 0;
        for (int dev = 0; dev < nDevices; dev++) {
        	if (std::find(disabled_gpus.begin(), disabled_gpus.end(), dev) == disabled_gpus.end()) {
        		jobs_possible_all += (int)((max_heap_size[dev] - mem_req) / mem_job);
        	}
        }
        int jobs_required = n * 2 * 8;
        std::cout << log_time() << " [INFO] BASE MEMORY REQUIRED: " << mem_req << " BYTES" << std::endl;
        std::cout << log_time() << " [INFO] MEMORY REQUIRED PER DYNEX CHIP: " << mem_job << " BYTES" << std::endl;
        std::cout << log_time() << " [INFO] MAX DYNEX CHIPS FITTING IN MEMORY (ALL GPUs): " << jobs_possible_all << std::endl;
        std::cout << log_time() << " [INFO] PARALLEL DYNEX CHIPS REQUIRED: " << jobs_required << std::endl;

        // num_jobs_all -> total jobs over all gpus:
        num_jobs_all = jobs_possible_all;
        if (jobs_possible_all > jobs_required) num_jobs_all = jobs_required;
        if (num_jobs_all > maximum_jobs) num_jobs_all = maximum_jobs; // user defined max #jobs
        std::cout << log_time() << " [INFO] ==> PREPARING " << num_jobs_all << " DYNEX CHIPS..." << std::endl;

        // loop through all GPUs:
        int num_jobs_free = num_jobs_all;
        
        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev; 
            gpuErrchk(cudaSetDevice(device_id));
            // only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		    // calculate number of jobs to be created => num_jobs[dev]:
		    int jobs_possible = 0;
		    jobs_possible = (int)((max_heap_size[dev] - mem_req) / mem_job);
		    // less jobs than space here?
		    if (jobs_possible > num_jobs_free) {
		        jobs_possible = num_jobs_free;
		        nDevices = dev + 1; // reduce number of devices needed
		    }
		    num_jobs[dev] = jobs_possible;
		    num_jobs_free = num_jobs_free - num_jobs[dev];

		    // test small cnf on multiple gpus:
		    /*nDevices = 2;
		    if (dev == 0) num_jobs[dev] = 4;
		    if (dev == 1) num_jobs[dev] = 8;
		    num_jobs_all = 12;
		    */// -> check passed

		    std::cout << log_time() << " [GPU " << device_id << "] PREPARING " << num_jobs[dev] << " DYNEX CHIPS..." << std::endl;
		
		    /// INIT MEMORY WITH KERNEL: ------------------------------------------------------------------------------------------
		    std::cout << log_time() << " [GPU " << device_id << "] ALLOCATING MEMORY... ";
		    // new allocate from host:
		    // job template:
		    job_struct job_template;
			    job_template.threadi = -1;
			    job_template.lambda = new int[n]; for (int i = 0; i < n; i++) job_template.lambda[i] = 0;
			    job_template.lambda_pos = 0;
			    job_template.lambda_bin = new bool[n * 2 + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.lambda_bin[i] = false;
			    job_template.complexity_counter = 0;
			    job_template.n = n;
			    job_template.m = m;
			    job_template.stage = 0;
			    job_template.polarity = false;
			    job_template.starting_Xk = -1;
			    job_template.Xk = -1;
			    job_template.header = new bool[n * 2 + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.header[i] = false;
			    job_template.isSat = true;
			    job_template.new_units = new int[2 * n + 1];
			    job_template.new_units_pos = 0;
			    job_template.new_units_bin = new bool[2 * n + 1]; for (int i = 0; i < (n * 2 + 1); i++) job_template.new_units_bin[i] = false;
			    job_template.max_adj_size = max_adj_size;
			    job_template.flipped = true;
			    job_struct d_job_template;
			    // create h_jobs and copy to d_jobs:
		    	int jobs_bytes = num_jobs[dev] * sizeof(job_template);
		    	h_jobs[dev] = (job_struct*)calloc((size_t) jobs_bytes, sizeof(size_t));
		    	for (int i = 0; i < num_jobs[dev]; i++) h_jobs[dev][i] = job_template;
		    	//copy jobs over to GPU (including sub arrays):
			    uint64_cu mem_reserved = 0;
			    for (int i = 0; i < num_jobs[dev]; i++) {
					gpuErrchk(cudaMalloc(&(h_jobs[dev][i].lambda), (n) * sizeof(int))); 
					gpuErrchk(cudaMalloc(&(h_jobs[dev][i].lambda_bin), (2 * n + 1) * sizeof(bool)));
					gpuErrchk(cudaMalloc(&(h_jobs[dev][i].header), (2 * n + 1) * sizeof(bool)));
					gpuErrchk(cudaMalloc(&(h_jobs[dev][i].new_units), (2 * n + 1) * sizeof(int)));
					gpuErrchk(cudaMalloc(&(h_jobs[dev][i].new_units_bin), (2 * n + 1) * sizeof(bool)));
					mem_reserved += (2 * n + 1) * sizeof(bool) * 3 + n * sizeof(int) + (2 * n + 1) * sizeof(int);
			    }
			    std::cout << " DONE." << std::endl;
			    std::cout << log_time() << " [GPU " << device_id << "] ALLOCATED " << mem_reserved << " BYTES.";
			    std::cout << std::endl;
			    gpuErrchk(cudaMalloc((void**)&d_jobs[dev], jobs_bytes)); //reserve memory for all jobs
			    gpuErrchk(cudaMemcpy(d_jobs[dev], h_jobs[dev] ,jobs_bytes, cudaMemcpyHostToDevice));
			    gpuErrchk(cudaMalloc((void**)&d_a[dev], m * sizeof(int))); // <== works
		    gpuErrchk(cudaMalloc((void**)&d_b[dev], m * sizeof(int))); // <== works
		    gpuErrchk(cudaMalloc((void**)&d_c[dev], m * sizeof(int))); // <== works
		    gpuErrchk(cudaMalloc((void**)&d_adj_opp[dev], max_adj_size * (2*n+1) * sizeof(int))); // <== works
		    gpuErrchk(cudaMalloc((void**)&d_adj_opp_sizes[dev], (2*n+1) * sizeof(int))); // <== works
		    
		    // d_jobs (not really needed but ensures that we can access d_jobs properly):
		    std::cout << log_time() << " [GPU " << device_id << "] INITIALIZING CHIPS... ";
		    init_dynex_jobs << <  num_jobs[dev] , 1 >> > (num_jobs[dev], n, m, max_adj_size, d_jobs[dev]); 
		    gpuErrchk(cudaDeviceSynchronize());
		    cudaError_t err = cudaGetLastError();
		    if (err != cudaSuccess) {
		        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		        return false;
		    }
		    std::cout << "DONE" << std::endl;

		    // d_a, d_b, d_c:
		    std::cout << log_time() << " [GPU " << device_id << "] COPYING PROBLEM... ";
		    int* d_a_tmp;
		    int* d_b_tmp;
		    int* d_c_tmp;
		    gpuErrchk(cudaMalloc((void**)&d_a_tmp, m * sizeof(int)));
		    gpuErrchk(cudaMalloc((void**)&d_b_tmp, m * sizeof(int)));
		    gpuErrchk(cudaMalloc((void**)&d_c_tmp, m * sizeof(int)));
		    gpuErrchk(cudaMemcpy(d_a_tmp, a, m * sizeof(int), cudaMemcpyHostToDevice));
		    gpuErrchk(cudaMemcpy(d_b_tmp, b, m * sizeof(int), cudaMemcpyHostToDevice));
		    gpuErrchk(cudaMemcpy(d_c_tmp, c, m * sizeof(int), cudaMemcpyHostToDevice));
		    init_dynex_abc << <  m , 1 >> > (m, d_a_tmp, d_b_tmp, d_c_tmp, d_a[dev], d_b[dev], d_c[dev] ); 
		    gpuErrchk(cudaDeviceSynchronize());
		    err = cudaGetLastError();
		    if (err != cudaSuccess) {
		        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		        return false;
		    }
		    std::cout << "DONE" << std::endl;

		    // d_adj_opp, d_adj_opp_sizes:
		    std::cout << log_time() << " [GPU " << device_id << "] COPYING CHIP TABLES... ";
		    int* d_adj_opp_tmp;
		    gpuErrchk(cudaMalloc((void**)&d_adj_opp_tmp, max_adj_size * (2 * n + 1) * sizeof(int)));
		    gpuErrchk(cudaMemcpy(d_adj_opp_tmp, adj_opp, max_adj_size* (2 * n + 1) * sizeof(int), cudaMemcpyHostToDevice));
		    int* d_adj_opp_sizes_tmp;
		    gpuErrchk(cudaMalloc((void**)&d_adj_opp_sizes_tmp, (2 * n + 1) * sizeof(int)));
		    gpuErrchk(cudaMemcpy(d_adj_opp_sizes_tmp, adj_opp_sizes, (2 * n + 1) * sizeof(int), cudaMemcpyHostToDevice));
		    init_dynex <<< 1,1 >>> (num_jobs[dev], n, m, max_adj_size, d_adj_opp_tmp, d_adj_opp_sizes_tmp, d_adj_opp[dev], d_adj_opp_sizes[dev]); // PREPARE d_jobs on GPU, d_a, d_b, d_c
		    gpuErrchk(cudaDeviceSynchronize());
		    err = cudaGetLastError();
		    if (err != cudaSuccess) {
		        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		        return false;
		    }
		    std::cout << "DONE" << std::endl;
		    std::cout << log_time() << " [GPU " << device_id << "] INITITALIZED." << std::endl;
		    /// --------------------------------------------------------------------------------------------------------------
            }
        }

        /// MALLOB: update_capacity -> let mallob know how many chips I can run ++++++++++++++++++++++++++++++++++++++++++
        if (!testing) {
		std::vector<std::string> p3;
		p3.push_back("network_id=" + MALLOB_NETWORK_ID);
		p3.push_back("job_id=" + std::to_string(JOB_ID));
		p3.push_back("capacity=" + std::to_string(num_jobs_all));
		jsonxx::Object o3 = mallob_mpi_command("update_capacity", p3);
		int num_updated = o3.get<jsonxx::Number>("RESULT");
		if (num_updated == 0) {
		    std::cout << log_time() << TEXT_RED << " [INFO] ERROR: Tried to update capacity on MALLOB, returned 0" << TEXT_DEFAULT << std::endl;
		    return false;
		}
		std::cout << log_time() << " [MALLOB] CAPACITY NOTIFICATION SENT" << std::endl;
        }
        /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	int _CHIP_FROM, _CHIP_TO, CHIPS_REQUIRED;
        /// MALLOB: retrieve from-to from mallob: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (!testing) {
		std::vector<std::string> p4;
		p4.push_back("network_id=" + MALLOB_NETWORK_ID);
		p4.push_back("version=" + VERSION);
		jsonxx::Object o4 = mallob_mpi_command("request_fromto", p4);
		CHIPS_REQUIRED = atoi(o4.get<jsonxx::String>("CHIPS").c_str());
		if (CHIPS_REQUIRED == 0) {
		    std::cout << log_time() << TEXT_RED << " [INFO] NO CHIPS REQUIRED FOR THE CURRENT JOB - PLEASE UPDATE YOUR DYNEXSOLVE VERSION" << TEXT_DEFAULT << std::endl;
		    return false;
		}
		_CHIP_FROM = atoi(o4.get<jsonxx::String>("CHIP_FROM").c_str());
		_CHIP_TO = atoi(o4.get<jsonxx::String>("CHIP_TO").c_str());
		std::cout << log_time() << " [MALLOB] WE GOT CHIPS " << _CHIP_FROM << " TO " << _CHIP_TO << " ASSIGNED." << std::endl;
		// did we get too many assigned?
		if (CHIPS_REQUIRED > num_jobs_all) {
		    std::cout << log_time() << TEXT_RED << " [INFO] TOO MANY CHIPS ASSIGNED TO ME, WANTED " << CHIPS_REQUIRED << ", WE HAVE " << num_jobs_all << std::endl;
		    return false;
		}
        }
        /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if (testing) {
            _CHIP_FROM = 0;
            _CHIP_TO = num_jobs_all;
            std::cout << log_time() << TEXT_GREEN << " [INFO] testing! set CHIP_FROM - CHIP_TO TO " << _CHIP_FROM << " to " << _CHIP_TO << TEXT_DEFAULT << std::endl;
        }

        // set CHIP_FROM, CHIP_TO for every GPU:
        int chips_pointer = _CHIP_FROM;
        for (int i = 0; i < nDevices; i++) {
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), i) == disabled_gpus.end()) {
		    CHIP_FROM[i] = chips_pointer;
		    CHIP_TO[i] = chips_pointer + num_jobs[i] - 1;
		    chips_pointer = chips_pointer + num_jobs[i];
		    std::cout << log_time() << " [GPU " << i << "] GOT CHIPS " << CHIP_FROM[i] << " TO " << CHIP_TO[i] << std::endl;
            }
        }

    } // --- work initiated ----------------------------------------------------------------------------------------------

    // configure threads and blocks: int* h_solved = (int*)calloc((size_t) 1, sizeof(int));
    int * threadsPerBlock = new int[nDevices];
    for (int i = 0; i < nDevices; i++) threadsPerBlock[i] = 1;

    int* numBlocks = new int[nDevices];
    for (int i = 0; i < nDevices; i++) numBlocks[i] = abs(num_jobs[i] / threadsPerBlock[i]);

    // -------------------------------------------------------------------------------------------------------------------------------------------------------------
    // start job (first badge):
    cudaEvent_t *start = new cudaEvent_t[nDevices];
    cudaEvent_t *stop  = new cudaEvent_t[nDevices];

    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
        	cudaEventCreate(&start[dev]);
        	cudaEventCreate(&stop[dev]);
        }
    }

    // peak performance -------------------------------------------------------------------------------------------------------------------------------
    float* miner_hashrate = new float[nDevices]; for (int i = 0; i < nDevices; i++) miner_hashrate[i] = 0;
    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
        	cudaEventCreate(&start[dev]);
        	cudaEventCreate(&stop[dev]);
        	std::cout << log_time() << " [GPU " << device_id << "] PEAK PERFORMANCE: " ;
	        gpuErrchk(cudaEventRecord(start[dev]));
	        test_flops<<<1,1>>>(10000000); 
	        gpuErrchk(cudaEventRecord(stop[dev]));
	        cudaError_t err = cudaGetLastError();
		    if (err != cudaSuccess) {
		        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		        return false;
		    }
	        gpuErrchk(cudaEventSynchronize(stop[dev]));
			float elapsed = 0;
			cudaEventElapsedTime(&elapsed, start[dev], stop[dev]);
			float rate = 10000000 / (elapsed/1000);
			std::cout << std::fixed << std::setprecision(0) << rate << " FLOPS" << std::endl; 
			miner_hashrate[dev] = (100.00 * (rate / (float)14777863 )) / (float)2942 * (float)num_jobs[dev];
			if (miner_hashrate[dev]<1) miner_hashrate[dev] = 1.0;
	    }
    }
    
    uint64_cu max_complexity = std::numeric_limits<uint64_t>::max(); //pow(n, 6)*nDevices;
    uint64_cu steps_per_run = steps_per_batch;
	std::cout << log_time() << " [INFO] ---------------------------------------------------------" << std::endl;
    std::cout << log_time() << TEXT_CYAN << " [INFO] STARTING " << num_jobs_all << " DYNEX CHIPS ON GPU..." << TEXT_DEFAULT << std::endl;
    
    /// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
    if (!testing) {
        std::vector<std::string> p5;
        p5.push_back("network_id=" + MALLOB_NETWORK_ID);
        p5.push_back("atomic_status=" + std::to_string(ATOMIC_STATUS_RUNNING));
        p5.push_back("steps_per_run=" + std::to_string(steps_per_run));
        p5.push_back("steps=" + std::to_string(h_total_steps_all));
        p5.push_back("version=" + VERSION);
        jsonxx::Object o5 = mallob_mpi_command("update_job_atomic", p5);
        if (o5.get<jsonxx::Number>("RESULT") == 0) { 
        	std::cout << TEXT_RED << " [INFO] ERROR: ATOMIC JOB NOT EXISTING OR EXPIRED" << TEXT_DEFAULT << std::endl;
        	std::cout << TEXT_RED << " PLEASE DELETE YOUR GPU_*.bin FILES, YOU CANNOT CONTINUE WORK FROM THERE ANYMORE." << TEXT_DEFAULT << std::endl;  
        	return false; 
        }
        std::cout << log_time() << " [MALLOB] ATOMIC STATE UPDATED" << std::endl;
    }
    /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    /// looped kernel start:
    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
        	gpuErrchk(cudaEventRecord(start[dev]));
        	run_DynexChipUpperBound << <numBlocks[dev], threadsPerBlock[dev] >> > (dev, n, m, max_adj_size, d_solved[dev], d_lambda_solution[dev], d_total_steps[dev], !work_in_progress, steps_per_run, CHIP_FROM[dev], CHIP_TO[dev], d_jobs[dev], d_a[dev], d_b[dev], d_c[dev], d_adj_opp[dev], d_adj_opp_sizes[dev]); // init only when work_in_progress = false
        }
    }
    for (int dev = 0; dev < nDevices; dev++) {
        if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
        // only not disabled gpus:
        if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaEventRecord(stop[dev]));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
		    printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		    return false;
		}
        }
    }

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // loop until solution found OR max complexity n^5 reached
    uint64_cu * rem_steps = (uint64_cu*)calloc((size_t)nDevices, sizeof(uint64_cu));
    for (int i=0; i<nDevices; i++) rem_steps[i] = 0;

    uint64_cu rem_steps_all = h_total_steps_all;
    int h_solved_all = 0;
    int h_solved_gpu = -1;
    float miner_hashrate_all;
    float miner_milliseconds_all = 60; //60s
    
    uint64_cu count_batches = 0;

    while (h_solved_all == 0 && h_total_steps_all < max_complexity && !dynex_quit_flag) {

		count_batches ++;

        float milliseconds;
        miner_hashrate_all = 0;
		float pool_hashrate_all = 0;

        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
	    	// only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
			    gpuErrchk(cudaMemcpy(h_total_steps, d_total_steps[dev], sizeof(uint64_cu), cudaMemcpyDeviceToHost));
			    h_total_steps_dev[dev] = h_total_steps[0];
			    gpuErrchk(cudaMemcpy(h_solved, d_solved[dev], sizeof(int), cudaMemcpyDeviceToHost));

			    if (h_solved[0] != 0) {
			        h_solved_all = h_solved[0];
			        h_solved_gpu = dev;
			    }

			    // continue run for steps_per_run runs:
			    gpuErrchk(cudaEventSynchronize(stop[dev]));
			    milliseconds = 0;
			    cudaEventElapsedTime(&milliseconds, start[dev], stop[dev]);
			    float hashrate = (h_total_steps[0] - rem_steps[dev]) / milliseconds * 1000;
			    uint64_cu steps_performed_this_batch = h_total_steps[0] - rem_steps[dev];
			    rem_steps[dev] = h_total_steps[0];
			    h_total_steps_all += steps_performed_this_batch;
			    uint64_cu average_steps = h_total_steps[0] / num_jobs[dev];
			    double ocompl = log(average_steps) / log(n);
			    miner_hashrate_all += miner_hashrate[dev];
			    float pool_hashrate = miner_hashrate[dev];
			    pool_hashrate_all += pool_hashrate;

			    auto t2 = std::chrono::high_resolution_clock::now();
			    float uptime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0/60.0/60.0;
	    		
			    std::cout << log_time() << TEXT_SILVER << " [GPU " << dev << "] " << h_total_steps[0] << " STEPS (+" << steps_performed_this_batch << ") | " << milliseconds / 1000 << "s | FLOPS = " << std::fixed << std::setprecision(3) << hashrate / 1000 << " kFLOPS | HR = " << std::setprecision(3) << pool_hashrate << " H | AVG(O)n ^ " << std::setprecision(5) << ocompl << " | CIRCUIT SIMULATION (" << std::fixed << std::setprecision(2) << uptime << "h)" << TEXT_DEFAULT << std::endl;
            }
        }
        
        // summary of all:
        float hashrate = (h_total_steps_all - rem_steps_all) / milliseconds * 1000;
        uint64_cu steps_performed_this_batch_all = h_total_steps_all - rem_steps_all;
        rem_steps_all = h_total_steps_all;
        uint64_cu average_steps = h_total_steps_all / num_jobs_all;
        double ocompl = log(average_steps) / log(n);
        
        if (use_multi_gpu) {
            std::cout << log_time() << TEXT_SILVER << " [GPU *] " << h_total_steps_all << " STEPS (+" << steps_performed_this_batch_all << ") | " << milliseconds / 1000 << "s | FLOPS = " << std::fixed << std::setprecision(3) << hashrate / 1000 << " kFLOPS | HR = " << std::setprecision(3) << miner_hashrate_all << " H | AVG(O)n ^ " << std::setprecision(5) << ocompl << TEXT_DEFAULT << std::endl;
        }
        
        // reset d_total_steps:
        h_total_steps[0] = 0;
        for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
			gpuErrchk(cudaMemcpy(d_total_steps[dev], h_total_steps, sizeof(uint64_cu), cudaMemcpyHostToDevice));
		}
	} 
        
        if (!testing) {
	    	/// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
	    	std::vector<std::string> p5;
	    	p5.push_back("network_id=" + MALLOB_NETWORK_ID);
	    	p5.push_back("atomic_status=" + std::to_string(ATOMIC_STATUS_RUNNING));
	    	p5.push_back("steps_per_run=" + std::to_string(steps_per_run));
	    	p5.push_back("steps=" + std::to_string(h_total_steps_all));
	    	p5.push_back("hr=" + std::to_string(int(hashrate / 1000)));
	    	p5.push_back("hradj=" + std::to_string(int(pool_hashrate_all)));
	    	p5.push_back("version=" + VERSION);
	    	jsonxx::Object o5 = mallob_mpi_command("update_job_atomic", p5);
	    	/*
	    	if (!o5.has<jsonxx::Boolean>("ERROR")) {
			    if (o5.get<jsonxx::Number>("RESULT")==0) {
				std::cout << TEXT_RED << " [INFO] ERROR: ATOMIC JOB NOT EXISTING" << TEXT_DEFAULT << std::endl;
				return false;
			    }
			    else {
				std::cout << log_time() << " [MALLOB] ATOMIC STATE UPDATED" << std::endl;
			    }
	    	}
	    	*/

	    	/// MALLOB: get network stats ++++++++++++++++++++++++++++++++++++++++++
	    	/*
			std::vector<std::string> ps;
			jsonxx::Object os = mallob_mpi_command("get_status", ps);
			if (!os.has<jsonxx::Boolean>("ERROR")) {
				std::string network_hr = os.get<jsonxx::String>("HR");
				std::string network_hr_adj = os.get<jsonxx::String>("HR_ADJ");
				std::cout << log_time() << " [MALLOB] INFO: NETWORK FLOPS = " << network_hr << " kFLOPS | NETWORK HR = " << network_hr_adj << " H" << std::endl;
			}
		*/
	    	/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        }
		/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		/// MALLOB: validate_job-> ask mallob if the job is still supposed to run +++++++++++++++++++++++++++++++++++++++++
		/* fix removed for now - unstable?
		    std::vector<std::string> p6;
		    p6.push_back("network_id="+MALLOB_NETWORK_ID);
		    jsonxx::Object o6 = mallob_mpi_command("validate_job", p6);	    
		    if (!o6.get<jsonxx::Boolean>("RESULT")) {std::cout << TEXT_RED << " [INFO] JOB WAS DETECTED TO HAVE ENDED" << TEXT_DEFAULT << std::endl; return false;}
		    std::cout << log_time() << " [MALLOB] JOB VALIDATED" << std::endl;
		*/
		/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	    uint64_cu leffom = miner_hashrate_all; 
		leffom = leffom / miner_milliseconds_all * (milliseconds / 1000) * 60; 
		if (!testing) dynexservice.leffom += leffom; 
	    /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
        /// looped kernel start:
        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
            // only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		    gpuErrchk(cudaEventRecord(start[dev]));
		    run_DynexChipUpperBound << <numBlocks[dev], threadsPerBlock[dev] >> > (dev, n, m, max_adj_size, d_solved[dev], d_lambda_solution[dev], d_total_steps[dev], false, steps_per_run, CHIP_FROM[dev], CHIP_TO[dev], d_jobs[dev], d_a[dev], d_b[dev], d_c[dev], d_adj_opp[dev], d_adj_opp_sizes[dev]);
            }
        }
        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev; gpuErrchk(cudaSetDevice(device_id));
            // only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
		    gpuErrchk(cudaDeviceSynchronize());
		    gpuErrchk(cudaEventRecord(stop[dev]));
		    cudaError_t err = cudaGetLastError();
		    if (err != cudaSuccess) {
		        printf(" [INFO] Error: %s\n", cudaGetErrorString(err));
		        return false;
		    }
            }
        }
    }
    if (dynex_quit_flag) std::cout << log_time() << " [INFO] QUIT REASON: dynex_quit_flag = 1 " << std::endl;
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    // stop dynexservice:    
    dynexservice.stop();
    
    // solution found?
    if (h_solved_all != 0) {
        std::cout << log_time() << TEXT_YELLOW;
        printf(" [GPU %d] SOLUTION FOUND - FINISHED WITH %llu TOTAL COMPUTATIONAL STEPS.\n",h_solved_gpu, h_total_steps_all);
        std::cout << TEXT_DEFAULT;
        //solution found:
        int* lambda_solution = (int*)calloc((size_t)n, sizeof(int));
        solution = (bool*)calloc((size_t)n, sizeof(bool));
        gpuErrchk(cudaSetDevice(h_solved_gpu)); // from the gpu which found the sol
        gpuErrchk(cudaMemcpy(lambda_solution, d_lambda_solution[h_solved_gpu], sizeof(int)* n, cudaMemcpyDeviceToHost));

        for (int j = 0; j < n; j++) {
            int var = lambda_solution[j];
            if (var > n) var = -(var - n);
            solution[abs(var) - 1] = true; if (var < 0) solution[abs(var) - 1] = false;
        }
        // verify solution:
        bool verify_sat = true;
        for (int j = 0; j < m; j++) {
            int lita = a[j]; bool a_pol = lita > 0 ? true : false;
            int litb = b[j]; bool b_pol = litb > 0 ? true : false;
            int litc = c[j]; bool c_pol = litc > 0 ? true : false;
            if (solution[abs(lita) - 1] != a_pol && solution[abs(litb) - 1] != b_pol && solution[abs(litc) - 1] != c_pol) {
            	std::cout << log_time() ;
                printf(" [INFO] ERROR: clause %d [%d %d %d] has assignment %d %d %d\n", j, lita, litb, litc, solution[abs(lita) - 1], solution[abs(litb) - 1], solution[abs(litc) - 1]);
                verify_sat = false;
                break;
            }
        }
        if (verify_sat) { 
#ifdef WIN32
            printf("c SOLUTION IS CERTIFIED.\n"); 
#else
            printf("%sc SOLUTION IS CERTIFIED.%s\n", TEXT_YELLOW, TEXT_DEFAULT);
#endif
        }
        else { 
#ifdef WIN32
            printf("c ERROR: SOLUTION NOT CERTIFIED.\n"); return false; 
#else
            printf("%sc ERROR: SOLUTION NOT CERTIFIED.%s\n", TEXT_RED, TEXT_DEFAULT); return false;
#endif
        }
        
        //write solution to file: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        std::string solfile = JOB_FILENAME + ".solution.txt";
        FILE *fs = fopen(solfile.c_str(), "w");
        for (int i=0; i<n; i++) {
            if (solution[i]) {
            	fprintf(fs,"%d, ",i+1);
            } else {
            	fprintf(fs,"%d, ",(i+1)*-1);
            }
        }
        fclose(fs);  
	    std::cout << log_time() << " [INFO] SOLUTION WRITTEN TO " << solfile << std::endl;

        /// SUBMIT SOLUTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (!testing) {
        	upload_file(solfile);
        	std::cout << log_time() << TEXT_YELLOW << " [INFO] SOLUTION SUBMITTED TO DYNEX " << TEXT_DEFAULT << std::endl;
    	}
        ///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if (!testing) {
            /// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
            std::vector<std::string> p5;
            p5.push_back("network_id=" + MALLOB_NETWORK_ID);
            p5.push_back("atomic_status=" + std::to_string(ATOMIC_STATUS_FINISHED_SOLVED));
            p5.push_back("steps_per_run=" + std::to_string(steps_per_run));
            p5.push_back("steps=" + std::to_string(h_total_steps_all));
            p5.push_back("hr=0");
            jsonxx::Object o5 = mallob_mpi_command("update_job_atomic", p5);
            if (o5.get<jsonxx::Number>("RESULT") == 0) { 
            	std::cout << TEXT_RED << " [INFO] ERROR: ATOMIC JOB NOT EXISTING OR EXPIRED" << TEXT_DEFAULT << std::endl; 
            	std::cout << TEXT_RED << " PLEASE DELETE YOUR GPU_*.bin FILES, YOU CANNOT CONTINUE WORK FROM THERE ANYMORE." << TEXT_DEFAULT << std::endl;
            	return false; 
            }
            std::cout << log_time() << " [MALLOB] ATOMIC STATE UPDATED" << std::endl;
            /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            /// save state to disk: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for (int dev = 0; dev < nDevices; dev++) {
                if (use_multi_gpu) device_id = dev; 
                // only not disabled gpus:
        	if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
                	write_states(device_id, num_jobs[device_id], CHIP_FROM[device_id], CHIP_TO[device_id], h_total_steps_dev[device_id]); // structure holding all job data);
                }
            }
            /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        }
	    
	    // output solution:
            std::cout << log_time() << TEXT_YELLOW;
            printf(" [INFO] SOLUTION OUTPUT FROM CHIP %d:\n", h_solved_all-1);
            printf("s ");
            for (int j = 0; j < n; j++) {
                if (solution[j]) {
                    printf("%d ", j + 1);
                }
                else {
                    printf("%d ", (j + 1) * -1);
                }
            }
            printf(" 0\n");
            std::cout << TEXT_DEFAULT;

    }
    else {
        std::cout << log_time();
        printf(" [INFO] UNKNOWN SOLUTION - STOPPED AT %llu TOTAL COMPUTATIONAL STEPS.\n", h_total_steps_all);
        /// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
        if (!testing) {
            std::vector<std::string> p5;
            p5.push_back("network_id=" + MALLOB_NETWORK_ID);
            p5.push_back("atomic_status=" + std::to_string(ATOMIC_STATUS_FINISHED_UNKNOWN));
            jsonxx::Object o5 = mallob_mpi_command("update_job_atomic", p5);
            if (o5.get<jsonxx::Number>("RESULT") == 0) { 
            	std::cout << TEXT_RED << " [INFO] ERROR: ATOMIC JOB NOT EXISTING OR EXPIRED" << TEXT_DEFAULT << std::endl; 
            	std::cout << TEXT_RED << " PLEASE DELETE YOUR GPU_*.bin FILES, YOU CANNOT CONTINUE WORK FROM THERE ANYMORE." << TEXT_DEFAULT << std::endl;
            	return false; 
            }
            std::cout << log_time() << " [MALLOB] ATOMIC STATE UPDATED" << std::endl;
            /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        }
        /// save state to disk: +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for (int dev = 0; dev < nDevices; dev++) {
            if (use_multi_gpu) device_id = dev;
            // only not disabled gpus:
            if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
            	write_states(device_id, num_jobs[device_id], CHIP_FROM[device_id], CHIP_TO[device_id], h_total_steps_dev[device_id]); 
            }
        }
        /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        

    }

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// command line handler
///////////////////////////////////////////////////////////////////////////////////////////////////////////
char* getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// graceful exit handler
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void signalHandler( int signum ) {
   cout << log_time() << " CTRL+C Interrupt signal (" << signum << ") received. Quitting gracefully...\n";
   
   // stop miners:
   dynexchip.dynex_quit_flag = true; // stop signal to GPU job manager and CPU jobs
   dynexservice.dynex_hasher_quit_flag = true;    // stop signal to Dynex hasher service

   // update mallob that we interruped:
   if (MALLOB_ACTIVE) {
   	/// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
	std::vector<std::string> p5;
	p5.push_back("network_id="+MALLOB_NETWORK_ID);
	p5.push_back("atomic_status="+std::to_string(ATOMIC_STATUS_INTERRUPTED));
	jsonxx::Object o5 = mallob_mpi_command("update_job_atomic", p5);	    
	std::cout << log_time() << TEXT_SILVER << " [INFO] MALLOB: ATOMIC JOB UPDATED." << TEXT_DEFAULT << std::endl;
	/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   }
   
   std::cout << log_time() << TEXT_SILVER << " [INFO] FINISHING UP WORK ON GPU..." << TEXT_DEFAULT << std::endl; fflush(stdout);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    curl_global_init(CURL_GLOBAL_DEFAULT);

    signal(SIGINT, signalHandler); 

    std::cout << log_time() << " [INFO] ---------------------------------------------------------" << std::endl;
    std::cout << log_time() << TEXT_SILVER << " [INFO] DynexSolve v" << VERSION << " | Meaningful Mining " << TEXT_DEFAULT << std::endl;
    std::cout << log_time() << " [INFO] ---------------------------------------------------------" << std::endl;

    // parse command line options:

    //help command?
    if (cmdOptionExists(argv, argv + argc, "-h"))
    {
        std::cout << "HELP" << std::endl;
        std::cout << "usage: dynexsolve -mining-address <WALLET ADDR> [options]" << std::endl;
        std::cout << std::endl;
        std::cout << "-mining-address <WALLET ADDR>    wallet address to receive the rewards" << std::endl;
        std::cout << "-daemon-host <HOST>              RCP host address of dynexd (default: localhost)" << std::endl;
        std::cout << "-daemon-port <PORT>              RCP port of dynexd (default: 18333)" << std::endl;
        
        std::cout << "-stratum-url <IP-ADDR>           IP address(!) of the Stratum pool" << std::endl;
        std::cout << "-stratum-port <PORT>             port of the Stratum pool" << std::endl;
        std::cout << "-stratum-paymentid <PAYMENT ID>  payment ID to add to wallet address" << std::endl;
        std::cout << "-stratum-password <PASSWORD>     Stratum password (f.e. child@worker1)" << std::endl;
        
        std::cout << "-no-cpu                          run no Dynex chips on CPU" << std::endl;
        std::cout << "-no-gpu                          run no Dynex chips on GPU (WARNING: MINING NOT POSSIBLE)" << std::endl;
        std::cout << "-mallob-endpoint <IP>            set the endpoint for the Dynex Malleable Load Balancer" << std::endl;
        std::cout << "-devices                         show GPU devices on this system" << std::endl;
        std::cout << "-deviceid <GPU NUM>              which GPU to use (default: 0 = first one) when using 1 GPU" << std::endl;
        std::cout << "-multi-gpu                       uses all GPUs in the system (default: off)" << std::endl;
        std::cout << "-disable-gpu <ID,ID,ID>          disable certain GPUs (check -devices for IDs) when using multi-gpu" << std::endl;
        std::cout << "-maximum-chips <JOBS>            set maximum number of parallel Dynex Chips to be run on GPU (default: INT_MAX)" << std::endl;        
        std::cout << "-steps-per-batch <STEPS>         set number of steps per batch (default: 10000, min 10000)" << std::endl;
        std::cout << "-start-from-job <JOB_NUM>        set the starting job number (default: 0)" << std::endl;
	std::cout << "-cpu-chips <INT>                 set number of CPU Dynex-Chips to run (default: 4)" << std::endl;
        std::cout << "-alpha <DOUBLE>                  set alpha value of ODE" << std::endl;
        std::cout << "-beta <DOUBLE>                   set beta value of ODE" << std::endl;
        std::cout << "-gamma <DOUBLE>                  set gamma value of ODE" << std::endl;
        std::cout << "-delta <DOUBLE>                  set detla value of ODE" << std::endl;
        std::cout << "-epsilon <DOUBLE>                set epsilon value of ODE" << std::endl;
        std::cout << "-zeta <DOUBLE>                   set zeta value of ODE" << std::endl;
        std::cout << "-init_dt <DOUBLE>                set initial dt value of ODE" << std::endl;
        std::cout << "-debug                           enable debugging output" << std::endl;
        std::cout << "-test <INPUTFILE>                test Dynex Chips locally" << std::endl;
        std::cout << "-mallob-debug                    Enables debugging of MPI" << std::endl;
        std::cout << "-h                               show help" << std::endl;
        return EXIT_SUCCESS;
    }
    
    //test?
    char* tf = getCmdOption(argv, argv + argc, "-test");
    if (tf) {
        testing = true;
        testing_file = tf;
        MINING_ADDRESS = "XwnV1b9sULyFvmW8NGQyndJGWkF9eE13XKobuGvHUS4QFRrKH7Ze8tRFM6kPeLjLHyfLWPoo7r8RJKyqpcGxZHk32f2avgT4t";
        std::cout << log_time() << TEXT_GREEN << " [INFO] TESTING ACTIVATED:  " << testing_file << TEXT_DEFAULT << std::endl;
    }
    
    //stratum
    char* surl = getCmdOption(argv, argv + argc, "-stratum-url");
    if (surl) {
        STRATUM_URL = surl;
        stratum = true;
        std::cout << log_time() << " [INFO] STRATUM PROTOCOL ENABLED " << std::endl;
        std::cout << log_time() << " [INFO] STRATUM URL SET TO " << STRATUM_URL << std::endl;
    }
    
    char* sport = getCmdOption(argv, argv + argc, "-stratum-port");
    if (sport) {
        STRATUM_PORT = sport;
        std::cout << log_time() << " [INFO] STRATUM PORT SET TO " << STRATUM_PORT << std::endl;
    }
    
    char* spay = getCmdOption(argv, argv + argc, "-stratum-paymentid");
    if (spay) {
        STRATUM_PAYMENT_ID = spay;
        std::cout << log_time() << " [INFO] STRATUM PAYMENT ID SET TO " << STRATUM_PAYMENT_ID << std::endl;
    }
    
    char* spass = getCmdOption(argv, argv + argc, "-stratum-password");
    if (spass) {
        STRATUM_PASSWORD = spass;
        std::cout << log_time() << " [INFO] STRATUM PASSWORD SET TO " << STRATUM_PASSWORD << std::endl;
    }
    
    
    //mining-address
    char* ma = getCmdOption(argv, argv + argc, "-mining-address");
    if (ma) {
        MINING_ADDRESS = ma;
        std::cout << log_time() << " [INFO] MINING ADDRESS SET TO " << MINING_ADDRESS << std::endl;
    }
    
    if (MINING_ADDRESS=="") {
    	std::cout << log_time() << TEXT_RED << " ERROR. WALLET ADDRESS NOT SPECIFIED." << TEXT_DEFAULT << std::endl;
        return EXIT_FAILURE;
    }
    
    //daemon-host
    char* dh = getCmdOption(argv, argv + argc, "-daemon-host");
    if (dh) {
        DAEMON_HOST = dh;
        std::cout << log_time() << " [INFO] OPTION daemon-host SET TO " << DAEMON_HOST << std::endl;
    }
    
    //daemon-port
    char* dp = getCmdOption(argv, argv + argc, "-daemon-port");
    if (dp) {
        DAEMON_PORT = dp;
        std::cout << log_time() << " [INFO] OPTION daemon-port SET TO " << DAEMON_PORT << std::endl;
    }
    
    //mallob endpoint?
    char* me = getCmdOption(argv, argv + argc, "-mallob-endpoint");
    if (me) {
        mallob_endpoint = me;
        std::cout << log_time() << " [INFO] OPTION mallob-endpoint SET TO " << mallob_endpoint << std::endl;
    }

    //debugger?
    bool dynex_debugger = false;
    if (cmdOptionExists(argv, argv + argc, "-debug")) {
        dynex_debugger = true;
        debug = dynex_debugger;
        std::cout << log_time() << " [INFO] OPTION debug ACTIVATED" << std::endl;
    }

    //mallob-debug?
    mallob_debug = false;
    if (cmdOptionExists(argv, argv + argc, "-mallob-debug")) {
        mallob_debug = true;
        std::cout << log_time() << " [INFO] OPTION mallob_debug ACTIVATED" << std::endl;
    }

    //- multi - gpu
    if (cmdOptionExists(argv, argv + argc, "-multi-gpu")) {
        use_multi_gpu = true;
        std::cout << log_time() << " [INFO] OPTION multi-gpu ACTIVATED" << std::endl;
    }

    //disable gpu?
    bool disable_gpu = false;
    if (cmdOptionExists(argv, argv + argc, "-no-gpu")) {
        disable_gpu = true;
        std::cout << log_time() << " [INFO] OPTION -no-gpu ACTIVATED - " << TEXT_RED << "WARNING: ONLY FOR TESTING - MINING WILL NOT BE PERFORMED" << TEXT_DEFAULT << std::endl;
    }
    
    //disable certain?
    char* dgp = getCmdOption(argv, argv + argc, "-disable-gpu");
    if (dgp) {
    	if (use_multi_gpu) {
        	std::string disable_gpus = dgp;
        	std::vector<std::string>disabled_gpus_str = split(disable_gpus,',');
        	for (int i=0; i<disabled_gpus_str.size(); i++) disabled_gpus.push_back(atoi(disabled_gpus_str[i].c_str()));
        	for (int i=0; i<disabled_gpus.size(); i++) 
        		std::cout << log_time() << TEXT_GREEN << " [INFO] OPTION -disable-gpu DISABLED GPU " << disabled_gpus[i] << TEXT_DEFAULT << std::endl;
        } else  {
        	std::cout << log_time() << TEXT_RED << " [ERROR] Option -disable-gpu cannot be used without option -multi-gpu" << TEXT_DEFAULT << std::endl;
        	return EXIT_FAILURE;
        }
    }

    //alpha, beta, gamma, delta, epsilon, zeta:
    double dmm_alpha = 0.00125; //5.0;
    double dmm_beta = 2.0; //20.0;
    double dmm_gamma = 0.25;
    double dmm_delta = 0.05;
    double dmm_epsilon = 0.1;
    double dmm_zeta = 0.1;
    double init_dt = 0.15;
    char* a = getCmdOption(argv, argv + argc, "-alpha");
    if (a) {
        dmm_alpha = atof(a);
        std::cout << log_time() << " [INFO] OPTION alpha SET TO " << dmm_alpha << std::endl;
    }
    char* b = getCmdOption(argv, argv + argc, "-beta");
    if (b) {
        dmm_beta = atof(b);
        std::cout << log_time() << " [INFO] OPTION beta SET TO " << dmm_beta << std::endl;
    }
    char* g = getCmdOption(argv, argv + argc, "-gamma");
    if (g) {
        dmm_gamma = atof(g);
        std::cout << log_time() << " [INFO] OPTION gamma SET TO " << dmm_gamma << std::endl;
    }
    char* d = getCmdOption(argv, argv + argc, "-delta");
    if (d) {
        dmm_delta = atof(d);
        std::cout << log_time() << " [INFO] OPTION delta SET TO " << dmm_delta << std::endl;
    }
    char* e = getCmdOption(argv, argv + argc, "-epsilon");
    if (e) {
        dmm_epsilon = atof(e);
        std::cout << log_time() << " [INFO] OPTION epsilon SET TO " << dmm_epsilon << std::endl;
    }
    char* z = getCmdOption(argv, argv + argc, "-zeta");
    if (z) {
        dmm_zeta = atof(z);
        std::cout << log_time() << " [INFO] OPTION zeta SET TO " << dmm_zeta << std::endl;
    }
    char* dt = getCmdOption(argv, argv + argc, "-init_dt");
    if (dt) {
        init_dt = atof(dt);
        std::cout << log_time() << " [INFO] OPTION init_dt SET TO " << init_dt << std::endl;
    }

    //query devices?
    if (cmdOptionExists(argv, argv + argc, "-devices"))
    {
        int devnum = query_devices();
        return EXIT_SUCCESS;
    }

    //cpu_chips?
    int cpu_chips = 4;
    char* rc = getCmdOption(argv, argv + argc, "-cpu-chips");
    if (rc) {
        cpu_chips = atoi(rc);
        std::cout << log_time() << " [INFO] OPTION cpu-chips SET TO " << cpu_chips << std::endl;
    }

    //disable cpu?
    if (cmdOptionExists(argv, argv + argc, "-no-cpu")) {
        cpu_chips = 0;
        std::cout << log_time() << " [INFO] OPTION -no-cpu ACTIVATED" << std::endl;
    }

    //start_from_job specified?
    int start_from_job = 0;
    char* sfj = getCmdOption(argv, argv + argc, "-start-from-job");
    if (sfj) {
        start_from_job = atoi(sfj);
        std::cout << log_time() << " [INFO] OPTION start-from-job SET TO " << start_from_job << std::endl;
    }

    //maximum_chips specified?
    int maximum_jobs = INT_MAX;
    char* mj = getCmdOption(argv, argv + argc, "-maximum-chips");
    if (mj) {
        maximum_jobs = atoi(mj);
        std::cout << log_time() << " [INFO] OPTION maximum-chips SET TO " << maximum_jobs << std::endl;
    }

    //maximum_jobs specified?
    int steps_per_batch = 10000;
    char* spb = getCmdOption(argv, argv + argc, "-steps-per-batch");
    if (spb) {
        steps_per_batch = atoi(spb);
        if (steps_per_batch < 10000) steps_per_batch = 10000;
        std::cout << log_time() << " [INFO] OPTION steps-per-batch SET TO " << steps_per_batch << std::endl;
    }

    //deviceid specified?
    int device_id = 0;
    char* did = getCmdOption(argv, argv + argc, "-deviceid");
    if (did) {
        device_id = atoi(did);
        std::cout << log_time() << " [INFO] OPTION deviceid SET TO " << device_id << std::endl;
    }
    
    // ------------------------------------ end command line parameters --------------------------------------------------------------------

    // single or multi gpu?:
    cudaGetDeviceCount(&nDevices);
    std::cout << log_time() << " [INFO] FOUND " << nDevices << " INSTALLED GPU(s)" << std::endl;
    if (!use_multi_gpu) {
        // single gpu:
        std::cout << log_time() << TEXT_SILVER << " [INFO] USING GPU DEVICE " << device_id << TEXT_DEFAULT << std::endl;
        nDevices = 1;
    }
    else {
        // multi gpu:
        std::cout << log_time() << TEXT_SILVER << " [INFO] MULTI-GPU ENABLED: FOUND " << nDevices << " GPUs" << TEXT_DEFAULT << std::endl;
    }

    // init global vars: 
    h_total_steps = (uint64_cu*)calloc((size_t) 1, sizeof(uint64_cu));
    h_total_steps[0] = 0;
    jobs_bytes = (int*)calloc((size_t)nDevices, sizeof(int));
    h_total_steps_init = (uint64_cu*)calloc((size_t) nDevices, sizeof(uint64_cu));
    for (int i = 0; i < nDevices; i++) h_total_steps_init[i] = 0;
    CHIP_FROM = (int*)calloc((size_t)nDevices, sizeof(int));
    for (int i = 0; i < nDevices; i++) CHIP_FROM[i] = 0;
    CHIP_TO = (int*)calloc((size_t)nDevices, sizeof(int));
    for (int i = 0; i < nDevices; i++) CHIP_TO[i] = 0;
    num_jobs = (int*)calloc((size_t)nDevices, sizeof(int));
    
    /// update mallob_endpoint (permanent redirect)
    if (!testing) {
	    std::cout << log_time() << " [INFO] CONNECTING TO DYNEX MALLOB ENDPOINT " << mallob_endpoint << "..."; fflush(stdout);
	    //mallob_endpoint = mallob_follow_url(mallob_endpoint); //get redirect url
	    std::cout << " OK" << std::endl;
	}
    ////
    
    /// MALLOB: get network stats ++++++++++++++++++++++++++++++++++++++++++
	if (!testing) {
	    std::vector<std::string> ps;
	    jsonxx::Object os = mallob_mpi_command("get_status", ps);
	    std::string network_hr = os.get<jsonxx::String>("HR");
	    std::string network_hr_adj = os.get<jsonxx::String>("HR_ADJ");
	    std::cout << log_time() << " [MALLOB] INFO: NETWORK FLOPS = " << network_hr << " kFLOPS | HR = " << network_hr_adj << " H" << std::endl;
	}
    /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    // Check if we have unfinished local state: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    bool work_in_progress=false;
    std::string FN; FN = "GPU_" + std::to_string(device_id) + ".bin";
    ifstream ifile(FN.c_str());
    if (ifile) {
    	ifile.close();
    	char _MALLOB_NETWORK_ID[64];
    	char _JOB_FILENAME[1024];
    	ifstream infile;
    	infile.open(FN.c_str(), std::ios::in | std::ios::binary);
    	infile.read((char*)&_MALLOB_NETWORK_ID, 64);
    	int filenamesize; 
	    infile.read((char*)&filenamesize, sizeof(filenamesize));
	    infile.read((char*)&_JOB_FILENAME, filenamesize);
    	infile.close();
    	MALLOB_NETWORK_ID.assign(_MALLOB_NETWORK_ID,64); 
    	JOB_FILENAME.assign(_JOB_FILENAME,filenamesize);
	    std::cout << log_time() << TEXT_SILVER << " [INFO] FOUND WORK IN PROGRESS (" << FN << ") " << TEXT_DEFAULT << std::endl;
	    std::cout << log_time() << TEXT_SILVER << " [INFO] WITH MALLOB_NETWORK_ID = " << MALLOB_NETWORK_ID << " (" << MALLOB_NETWORK_ID.size() << ")" << TEXT_DEFAULT << std::endl;
	    std::cout << log_time() << TEXT_SILVER << " [INFO] WITH FILENAME = " << JOB_FILENAME << TEXT_DEFAULT << TEXT_DEFAULT << std::endl;

	    // still valid?
	    /// MALLOB: validate_job-> ask mallob if the job is still supposed to run +++++++++++++++++++++++++++++++++++++++++
	    if (!testing) {
		    std::vector<std::string> p6;
		    p6.push_back("network_id="+MALLOB_NETWORK_ID);
		    jsonxx::Object o6 = mallob_mpi_command("validate_job", p6);	 
		    if (o6.has<jsonxx::Boolean>("RESULT")) {   
			    if (!o6.get<jsonxx::Boolean>("RESULT")) {
				    std::cout << log_time() << TEXT_RED << " [MALLOB] JOB ALREADY ENDED, WE FIND NEW WORK" << TEXT_DEFAULT << std::endl; 
			    } else {
				    std::cout << log_time() << TEXT_SILVER << " [MALLOB] JOB IS STILL VALID, WE WILL CONTINUE WORKING ON IT..." << TEXT_DEFAULT << std::endl;
				    work_in_progress = true;
			    }
			} else {
				std::cout << log_time() << TEXT_RED << " [MALLOB] ERROR - YOU HAVE CURRENT WORK, BUT THIS EXPIRED. PLEASE DELETE ALL GPU_x.bin FILES" << TEXT_DEFAULT << std::endl;
				return EXIT_FAILURE;
			}
	    }
	    /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	    MALLOB_ACTIVE = true;
    }
    
    /// MALLOB ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // no work in progress, we need to sign up:
    if (!work_in_progress) {
    	// set random seed (making sure every restart sets same NETWORK_ID)
        srand((unsigned)time(NULL)); //fix similar random seed    srand(42);
	    // create a 32 byte unique identifier:
	    std::stringstream sstream;
        for (int i = 0; i < 32; i++) {
            int randbyte = rand() % 256;
            sstream << std::setw(2) << std::setfill('0') << std::hex << randbyte;
        }
	    MALLOB_NETWORK_ID = sstream.str();
	    std::cout << log_time() << " [MALLOB] UNIQUE MALLOB NETWORK_ID: "<< MALLOB_NETWORK_ID << " (" << MALLOB_NETWORK_ID.size() << ")" << std::endl;

	    // Register as new worker:
	    if (!testing) {
		    std::vector<std::string> p1;
		    p1.push_back("network_id="+MALLOB_NETWORK_ID); 
		    jsonxx::Object o1 = mallob_mpi_command("register_as_worker", p1);
		    bool r1 = o1.get<jsonxx::Boolean>("RESULT");
		    if (!r1) {
				std::cout << log_time() << TEXT_RED << " [MALLOB] REGISTERED AS NEW WORKER FAILED." << TEXT_DEFAULT << std::endl;
				return EXIT_FAILURE;
		    }
		    std::cout << log_time() << TEXT_SILVER << " [MALLOB] REGISTERED AS NEW WORKER: SUCCESS." << TEXT_DEFAULT << std::endl;
		    MALLOB_ACTIVE = true;
		    
		    // get work:
		    std::vector<std::string> p2;
		    jsonxx::Object o2 = mallob_mpi_command("get_work", p2);
		    JOB_ID = atoi(o2.get<jsonxx::String>("JOB_ID").c_str());
		    int CHIPS_AVAILABLE = atoi(o2.get<jsonxx::String>("CHIPS_AVAILABLE").c_str());
		    int CHIPS_REQUIRED = atoi(o2.get<jsonxx::String>("CHIPS_REQUIRED").c_str());
		    JOB_FILENAME = o2.get<jsonxx::String>("JOB_FILENAME"); //JOB_FILENAME = DATA_DIR+"/"+JOB_FILENAME;
		    double JOB_FEE = atof(o2.get<jsonxx::String>("JOB_FEE").c_str());
		    double JOB_SOLUTION_REWARD = atof(o2.get<jsonxx::String>("JOB_SOLUTION_REWARD").c_str());
		    std::cout << log_time() << TEXT_SILVER;
		    std::cout << " [MALLOB] JOB RECEIVED" << std::endl;
		    std::cout << TEXT_DEFAULT << log_time() << TEXT_SILVER << " [MALLOB] JOB_ID              : " << JOB_ID << std::endl;
		    std::cout << TEXT_DEFAULT << log_time() << TEXT_SILVER << " [MALLOB] CHIPS AVAILABLE     : " << CHIPS_AVAILABLE << "/" << CHIPS_REQUIRED << std::endl;
		    std::cout << TEXT_DEFAULT << log_time() << TEXT_SILVER << " [MALLOB] JOB FILENAME        : " << JOB_FILENAME << std::endl;
		    std::cout << TEXT_DEFAULT << log_time() << TEXT_SILVER << " [MALLOB] JOB FEE             : BLOCK REWARD + " << JOB_FEE <<  " DNX" << std::endl;
		    std::cout << TEXT_DEFAULT << log_time() << TEXT_SILVER << " [MALLOB] JOB SOLUTION REWARD : " << JOB_SOLUTION_REWARD <<  " DNX" << std::endl;
		    std::cout << TEXT_DEFAULT;
		    
		    // double check; chips also available?
		    if (CHIPS_AVAILABLE==0) {
		    	std::cout << log_time() << TEXT_RED << " [MALLOB] NO JOBS AVAILABLE" << TEXT_DEFAULT << std::endl;
		    	return EXIT_FAILURE;
		    }    
	    }
	    
	
    } // end -no work in progress
    
    // sanity check: mallob_network_id 64 bytes?
    if (MALLOB_NETWORK_ID.size()!=64) {
    	std::cout << log_time() << TEXT_RED << " [MALLOB] ERROR: MALLOB NETWORK ID HAS THE WRONG SIZE. ABORT." << TEXT_DEFAULT << std::endl;
    	return EXIT_FAILURE;
    }

    // testing?
    if (testing) JOB_FILENAME = testing_file;

    // file existing?
    if (!file_exists(JOB_FILENAME)) {
    	std::cout << log_time() << " [MALLOB] " << JOB_FILENAME << " NEEDS TO BE DOWNLOADED..." << std::endl;
    	if (!download_file(JOB_FILENAME)) return EXIT_FAILURE;
    	std::cout << log_time() << " [MALLOB] FILE SUCCESSFULLY DOWNLOADED." << std::endl;
    }
    
    /// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    // first step: check DAEMON connection
    if (!testing && !stratum) {
	    if (!stop_miner()) {
	    	std::cout << log_time() << TEXT_RED << " [INFO] ERROR: NO CONNECTION TO DYNEXD DAEMON" << TEXT_DEFAULT << std::endl;
	    } else {
	    	std::cout << log_time() << " [INFO] CONNECTION TO DYNEXD DAEMON SUCCESSFULLY ESTABLISHED" << std::endl;
	    }
	}

    //if (!disable_gpu && !load_cnf(argv[1])) {
    if (!disable_gpu && !load_cnf(JOB_FILENAME.c_str())) {
        return EXIT_FAILURE;
    }
    std::cout << log_time() << " [INFO] FORMULATION LOADED." << std::endl;

    // run CPU dynex chips
    bool dnxret = dynexchip.start(cpu_chips, JOB_FILENAME, dmm_alpha, dmm_beta, dmm_gamma, dmm_delta, dmm_epsilon, dmm_zeta, init_dt, dynex_debugger, steps_per_batch);
    if (disable_gpu) {
        while (!dynexchip.dynex_quit_flag) {;}
        return EXIT_SUCCESS;
    }
    
    // run GPU dynex chips:
    
    if (!run_dynexsolve(start_from_job, maximum_jobs, steps_per_batch, device_id, dynexchip.dynex_quit_flag, work_in_progress)) {
        std::cout << log_time() << " [INFO] EXIT WITH ERROR." << std::endl;
        return EXIT_FAILURE;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << log_time() << " [INFO] WALL TIME: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << "ms" << std::endl;
    
    curl_global_cleanup();

    std::cout << log_time() << " GOOD BYE." << std::endl;

    return EXIT_SUCCESS;
}
