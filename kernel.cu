// ----------------------------------------------------------------------------------------------------
// DYNEXSOLVE
// ----------------------------------------------------------------------------------------------------
// Copyright (c) 2021-2023, The Dynex Project
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

// for cuda:
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <assert.h>
#include <iomanip>

// for curl:
#include <curl/curl.h> 
#include "jsonxx.h"    
CURL* curl; // init curl

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

// defines and version:
std::string VERSION = "2.2.5";
std::string REVISION = "b";
//#define GPUDEBUG
#define MAX_NUM_GPUS      	 			32   // maximum supported by nvidia driver
#define HASHRATE_INTERVAL    			20
#define MAX_ATOMIC_ERR       			10
#define MAX_MALLOB_ERR       			5   
#define ADJ_DEFAULT          			1.3
#define MAX_REJECTED_SERIES  			20
#define PRECISION 						double //precision of ODE integration;
#define JOB_TYPE_SAT                    0
#define JOB_TYPE_MILP                   1
#define JOB_TYPE_QUBO                   2
#define JOB_TYPE_MAXSAT                 3
#define JOB_TYPE_FEDERATED_ML           4
#define JOB_TYPE_PRETRAINING_ML         5
#define JOB_TYPE_SUBSET_SUM             6
#define JOB_TYPE_INTEGER_FACTORISATION  7
#define ATOMIC_STATUS_ASSIGNED          0
#define ATOMIC_STATUS_RUNNING           1
#define ATOMIC_STATUS_FINISHED_SOLVED   2
#define ATOMIC_STATUS_FINISHED_UNKNOWN  3
#define ATOMIC_STATUS_INTERRUPTED       4

using namespace std;

#include "log.hpp" 			// logger
#include "picosha3.h" 		// sha3
#include "dynexsolve.hpp" 	// mallob communication handler class * REDACTED *
#include "Dynexchip.cpp" 	// cpu dynex chips class
#include "dynexservice.cpp" // stratum communication handler class * REDACTED *

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;

enum {
	ASSIGNED = 		0,
	RUNNING = 		1,
	SOLVED = 		2,
	UNKNOWN = 		3,
	CANCELLED = 	4
} ATOMIC_STATUS;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// global variables:
int n; 							// number of variables
int m; 							// number of clauses
bool* solution; 				// will contain the solution if found (size = n)
int nDevices = 0; 				// number of cuda devices
bool use_multi_gpu = false;		// single/multi GPU usage
std::string JOB_FILENAME = "";  
int num_jobs[MAX_NUM_GPUS] = {0};
int num_jobs_all = 0;

// job data structure:
typedef struct
{
	int threadi;
	int n;
	int m;
	uint32_t xl_max;
	int global;
	int loc;
	int solved; 
	int padding;
	int* cls;
	PRECISION* initial_conditions;
	PRECISION* x;
	PRECISION* dxdt;
	PRECISION t;
	PRECISION stepsize;
	PRECISION dmm_alpha;
	PRECISION dmm_beta;
	PRECISION dmm_gamma;
	PRECISION dmm_delta;
	PRECISION dmm_epsilon;
	PRECISION dmm_zeta;
	PRECISION energy;
} job_struct_2;

typedef struct {
	int loc;
	PRECISION energy;
	uint64_cu steps;
	bool solution[1];
} state_struct;

// vars:
int* cls{};
int* d_cls[MAX_NUM_GPUS]{};
state_struct* d_state[MAX_NUM_GPUS]{};
job_struct_2* d_jobs_2[MAX_NUM_GPUS]{};
job_struct_2* h_jobs_2[MAX_NUM_GPUS]{};
size_t max_heap_size[MAX_NUM_GPUS] = {0};
PRECISION dmm_alpha = 5.0;
PRECISION dmm_beta = 20.0;
PRECISION dmm_gamma = 0.25;
PRECISION dmm_delta = 0.05;
PRECISION dmm_epsilon = 0.1;
PRECISION dmm_zeta = 0.1;
PRECISION init_dt = 0.15;
int CNF_CHECKSUM = 0;
std::string CNF_DOWNLOADURL = "";
std::string CNF_SOLUTIONURL = "";
std::string CNF_SOLUTIONUSER = "";
std::atomic<int> overall_loc{0};
std::atomic<PRECISION> overall_energy{0};
float overall_hashrates[MAX_NUM_GPUS]{0};
int threadsPerBlock[MAX_NUM_GPUS];
int numBlocks[MAX_NUM_GPUS];
std::atomic<bool> atomic_updated{0};
PRECISION factor = 0.0;

// system definitions:
#define MAX_LIT_SYSTEM 3
int max_adj_size = 0;
uint64_cu PARALLEL_RUNS;
bool debug = false;
bool testing = false;
std::string testing_file;
bool DISC_OPERATION = false;
std::vector<int> disabled_gpus;
auto t0 = std::chrono::steady_clock::now();

// mallob definitions:
bool MALLOB_ACTIVE = false;
int JOB_ID = -1; 
std::string MALLOB_NETWORK_ID = "";

// default parameters:
std::string MINING_ADDRESS = ""; 
float rem_hashrate = 0;
int INTENSITY = 0;
float ADJ[MAX_NUM_GPUS] = {0};
bool SKIP = false;
std::string STATS = "";
std::string BUSID = "";
uint8_t network_id[32] = {0};

// stratum
bool stratum = false;
std::string STRATUM_URL = ""; 
int STRATUM_PORT = 0;  
std::string STRATUM_PAYMENT_ID = ""; 
std::string STRATUM_PASSWORD  = ""; 
int  STRATUM_DIFF = 0;

// Dynex classes:
std::atomic_bool dynex_quit_flag;
Dynex::dynexchip dynexchip;
Dynexservice::dynexservice dynexservice;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// system helper functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string log_time() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
	auto str = oss.str();
	return str;
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

template <typename T>
T atomic_fetch_min(atomic<T>* pv, typename atomic<T>::value_type v) noexcept {
	auto t = pv->load(std::memory_order_relaxed);
	while (std::min(v, t) != t) {
		if (pv->compare_exchange_weak(t, v, std::memory_order_relaxed, std::memory_order_relaxed))
			break;
	}
	return t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// FTP FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////
static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *stream)
{
	std::size_t retcode = std::fread( ptr, size, nmemb, static_cast<std::FILE*>(stream) );
	return retcode;
}

struct FtpFile {
	const char *filename;
	FILE *stream;
};

static int my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
	struct FtpFile *out = (struct FtpFile *)stream;
	if(!out->stream) {
		out->stream = fopen(out->filename, "wb");
		if(!out->stream)
			return 0;
	}
	return fwrite(buffer, size, nmemb, out->stream);
}

bool upload_file(const std::string filename) {
	CURLcode res;
	FILE *hd_src;
	struct stat file_info;
	unsigned long fsize;
	bool ret = false;

	/* get the file size of the local file */
	if(stat(filename.c_str(), &file_info)) {
		LogTS(TEXT_BRED) << "[ERROR] Couldn't open '" << filename.c_str() << "': " <<  strerror(errno) << std::endl;
		return false;
	}
	fsize = (unsigned long)file_info.st_size;

	LogTS() << "[INFO] Local file size: " << fsize << " bytes" << std::endl;

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
		std::string remoteurl = CNF_SOLUTIONURL + filename;
		curl_easy_setopt(curl, CURLOPT_URL, remoteurl.c_str());
		curl_easy_setopt(curl, CURLOPT_USERPWD, CNF_SOLUTIONUSER.c_str());
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
		if(res != CURLE_OK) {
			LogTS(TEXT_BRED) << "[ERROR] UPLOAD FAILED: " << curl_easy_strerror(res) << std::endl;
		} else {
			ret = true;
		}
		/* always cleanup */
		curl_easy_cleanup(curl);
	}
	fclose(hd_src); /* close the local file */
	return ret;
}

bool download_file(const std::string filename) {
	CURLcode res;

	struct FtpFile ftpfile = {
		filename.c_str(),
		NULL
	};

	curl = curl_easy_init();
	if (curl) {
		std::string remoteurl = "https://jobs.dynexcoin.org/" + filename; 
		if (debug) LogTS() << "[INFO] Downloading " << remoteurl << "..." << std::endl;
		curl_easy_setopt(curl, CURLOPT_URL, remoteurl.c_str());
		curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpfile);
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
		res = curl_easy_perform(curl);

		long status_code = 0;
		curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
		curl_easy_cleanup(curl);

		if (ftpfile.stream) {
			fclose(ftpfile.stream);
			if (res == CURLE_OK && status_code == 200) return true;
			std::remove(filename.c_str()); // delete broken file
		}
		if (CURLE_OK != res) {
			LogTS(TEXT_BRED) << "[ERROR] " << curl_easy_strerror(res) << std::endl;
			return false;
		}
		if (status_code != 200) {
			LogTS(TEXT_BRED) << "[ERROR] HTTP STATUS CODE: " << status_code << std::endl;
			return false;
		}
	}
	LogTS(TEXT_BRED) << "[ERROR] CURL FAILED" << std::endl;
	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stdout, " GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
bool load_cnf(const std::string& filename, const std::string& filehash = {}) {
	n = 0;
	m = 0;
	int checksum = 0;
	std::string line;
	auto sha3_256 = picosha3::get_sha3_generator<256>();
	auto t1 = std::chrono::steady_clock::now();

	std::ifstream file(filename);
	if (!file.is_open()) {
		LogTS(TEXT_BRED) << "[ERROR] FILE NOT FOUND: " << filename << std::endl;
		return false;
	}

	LogTS() << "[INFO] LOADING FILE: " << filename << std::endl;

	while (std::getline(file, line)) {
		sha3_256.process(line.cbegin(), line.cend());
		if (std::sscanf(line.c_str(), "p cnf %u %u", &n, &m) == 2) {
			break;
		}
	}

	if (!n || !m) {
		LogTS(TEXT_BRED) << "[ERROR] INVALID FORMAT" << std::endl;
		return false;
	}

	LogTS() << "[INFO] VARIABLES : " << n << std::endl;
	LogTS() << "[INFO] CLAUSES   : " << m << std::endl;
	LogTS() << "[INFO] RATIO     : " << ((double)m / n) << std::endl;

	/// reserve  memory:
	cls = (int*)calloc((size_t)m * MAX_LIT_SYSTEM, sizeof(int));

	// read CNF:
	int res[MAX_LIT_SYSTEM+1];
	int lit;
	int i = -1;
	while (std::getline(file, line)) {
		sha3_256.process(line.cbegin(), line.cend());

		lit = std::sscanf(line.c_str(), "%d %d %d %d", &res[0], &res[1], &res[2], &res[3]); // MAX_LIT_SYSTEM + 1
		if (lit == 0) continue; // skip comments and empty lines
		i++;
		if (i == m) break;

		// check max amount
		if (lit > MAX_LIT_SYSTEM && res[MAX_LIT_SYSTEM] != 0) {
			LogRTS(TEXT_BRED) << "[ERROR] CLAUSE " << i << " HAS " << lit << " LITERALS (" << MAX_LIT_SYSTEM << " ALLOWED)" << std::endl;
			return false;
		}

		for (int j = lit; j > 0; j--) {
			if (res[j-1] == 0) {
				lit--;
			} else if (res[j-1] > n || res[j-1] < -n) {
				LogRTS(TEXT_BRED) << "[INFO] CLAUSE " << i << " HAS BAD LITERAL " << res[j-1] << " (" << n << " ALLOWED)" << std::endl;
				return false;
			}
		}

		// do not allow zero
		if (lit == 0 || res[0] == 0) {
			LogRTS() << "[INFO] CLAUSE: " << i << " HAS NO LITERALS" << std::endl;
			return false;
		}

		if (debug && i % 100000 == 0) {
			LogRTS() << "[INFO] LOADING   : " << int(100.0 * (i + 1) / m) << "% " << std::flush;
		}

		for (int j = 0; j < MAX_LIT_SYSTEM; j++) {
			if (j >= lit) res[j] = res[j-1];
			cls[i * MAX_LIT_SYSTEM + j] = res[j];
			checksum += res[j];
		}
	}

	file.close();

	sha3_256.finish();
	std::array<uint8_t, picosha3::bits_to_bytes(256)> hash{};
	sha3_256.get_hash_bytes(hash.begin(), hash.end());
	std::string sha3hash = picosha3::bytes_to_hex_string(hash); // cat job.cnf | tr -d '\n' | rhash --sha3-256 -
	
	if (debug) {
		LogRTS() << "[INFO] LOADING   : " << int(100.0 * (i + 1) / m) << "% " << std::endl;
	}

	if (i + 1 != m) {
		LogRTS(TEXT_BRED) << "[ERROR] UNEXPECTED END OF FILE: " << i << " (" << m << " EXPECTED)" << std::endl;
		return false;
	}

	auto t2 = std::chrono::steady_clock::now();
	float dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	if (debug) LogTS() << "[INFO] LOADED IN " << dur << "ms" << std::endl;

	if (debug) {
		LogTS(TEXT_GRAY) << "[INFO] FIRST 10 CLAUSES:" << std::endl;
		for (i = 0; i < 10; i++) {
			LogTS(TEXT_GRAY) << "[INFO] CLAUSE " << i << ": ";
			for (int j = 0; j < MAX_LIT_SYSTEM; j++) {
				Log(TEXT_GRAY) << cls[i * MAX_LIT_SYSTEM + j] << " ";
			}
			Log() << std::endl;
		}
	}

	// verification with checksum:
	if (!testing) {
		if (filehash != sha3hash) {
			LogTS(TEXT_BRED) << "[ERROR] INCORRECT PROBLEM FILE" << std::endl;
			return false;
		}
	} else {
		checksum = checksum * m / n;
		if (debug) LogTS(TEXT_BCYAN) << "[INFO] " << sha3hash << " | " << checksum << std::endl;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Query devices
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int query_devices(int device_id) {
	int nDevices;
	if (cudaGetDeviceCount(&nDevices) != cudaSuccess) {
		LogTS(TEXT_BRED) << "[ERROR] Unable to get GPU amount" << std::endl;
		return 0;
	}
	int runtimeVersion;
	cudaRuntimeGetVersion(&runtimeVersion);
	int driverVersion;
	cudaDriverGetVersion(&driverVersion);
	LogTS() << "[INFO] CUDA RUNTIME: " << runtimeVersion/1000 << "." << runtimeVersion%1000/10 << std::endl;
	LogTS() << "[INFO] CUDA DRIVER:  " << driverVersion/1000 << "." << driverVersion%1000/10 << std::endl;
	LogTS() << "[INFO] FOUND " << nDevices << " INSTALLED GPU(s)" << std::endl;

	if (device_id >= 0 && device_id < nDevices) LogTS(TEXT_SILVER) << "[INFO] USING GPU DEVICE " << device_id << std::endl;

	BUSID = "";
	std::string adj_str;
	for (int i = (device_id==-1?0:device_id); i < nDevices; i++) {
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), i) != disabled_gpus.end()) continue; // skip disabled
		cudaSetDevice(i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		adj_str = "";
		if (ADJ[i]) {
			adj_str = " ADJ = ";
			adj_str.append(std::to_string(ADJ[i]));
		}
		LogTS(TEXT_BCYAN) << "[GPU " << i << "] " << std::setfill('0') << std::setw(2) << std::hex << devProp.pciBusID << ":"
			<< std::setw(2) << devProp.pciDeviceID << " " << devProp.name << " " << std::dec << devProp.totalGlobalMem/1024/1024 << " MB ("
			<< devProp.major << "." << devProp.minor << ")" << adj_str.c_str() << std::endl;

		BUSID.append(BUSID == "" ? "[" : ",").append(std::to_string(devProp.pciBusID));
		if (device_id != -1) break;
	}
	if (BUSID != "") BUSID.append("]");
	return nDevices;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void gpu_set_initial_conditions(const int dev, const int n, const int m, const int numchips, job_struct_2* d_jobs, state_struct* d_state,
	const PRECISION stepsize, const PRECISION dmm_alpha, const PRECISION dmm_beta, const PRECISION dmm_gamma, const PRECISION dmm_delta,
	const PRECISION dmm_epsilon, const PRECISION dmm_zeta) {

	for (int threadi = blockIdx.x * blockDim.x + threadIdx.x; threadi < numchips; threadi += blockDim.x * gridDim.x) {
		// globals
		if (threadi == 0) {
			d_state[0].steps = 0;
			d_state[0].energy = (PRECISION)m;
			d_state[0].loc = m;
			for (int i = 0; i < n; i++) {
				d_state[0].solution[i] = 0;
			}
		} else {
			for (int i = 0; i < m*3; i++) {
				d_jobs[threadi].cls[i] = d_jobs[0].cls[i]; // copy from thread 0
			}
		}

		for (int i = 0; i < n + m*2; i++) {
			d_jobs[threadi].dxdt[i] = 0.0;
			d_jobs[threadi].x[i] = 0.0;
		}

		// init job:
		d_jobs[threadi].t = 0.0;
		d_jobs[threadi].threadi = threadi;
		d_jobs[threadi].n = n;
		d_jobs[threadi].m = m;
		d_jobs[threadi].xl_max = 10000 * m;
		d_jobs[threadi].global = m;
		d_jobs[threadi].loc = m;
		d_jobs[threadi].energy = (PRECISION)m;
		d_jobs[threadi].solved = false;
		d_jobs[threadi].stepsize = stepsize; 
		d_jobs[threadi].dmm_alpha = dmm_alpha * (0.5 + (numchips - threadi + 1)/(PRECISION)numchips) ; //alpha distribution
		d_jobs[threadi].dmm_beta = dmm_beta;
		d_jobs[threadi].dmm_gamma = dmm_gamma;
		d_jobs[threadi].dmm_delta = dmm_delta;
		d_jobs[threadi].dmm_epsilon = dmm_epsilon;
		d_jobs[threadi].dmm_zeta = dmm_zeta;
	}
	
}
//------------------------------------------------------------------------------------------------------------------------------

__global__
void gpu_load_initial_conditions(const int dev, const int numchips, job_struct_2* d_jobs, bool use_random, bool testing) {
	for (int threadi = blockIdx.x * blockDim.x + threadIdx.x; threadi < numchips; threadi += blockDim.x * gridDim.x) {
		if (use_random) {
			// assign random initial states for Voltages:
			curandState state;
			curand_init((unsigned long long)clock(), threadi, 0, &state);
			for (int i = 0; i < d_jobs[threadi].n; i++) {
				float RANDOM = curand_uniform( &state );

				// three states:
				d_jobs[threadi].initial_conditions[i] = 0.0;
				if (RANDOM <= 0.33) d_jobs[threadi].initial_conditions[i] = -1.0;
				if (RANDOM >= 0.66) d_jobs[threadi].initial_conditions[i] = +1.0;
				
				// default 3 lanes:
				if (threadi == 0) d_jobs[threadi].initial_conditions[i] = 0.0;
				if (threadi == 1) d_jobs[threadi].initial_conditions[i] = 1.0;
				if (threadi == 2) d_jobs[threadi].initial_conditions[i] = -1.0;
				
			}
			// set Xs:
			for (int i=d_jobs[threadi].n; i<d_jobs[threadi].n + d_jobs[threadi].m; i++) {
				d_jobs[threadi].x[i] = 0.0;
			}
			// set Xl:
			for (int i=d_jobs[threadi].n+d_jobs[threadi].m; i<d_jobs[threadi].n + d_jobs[threadi].m*2; i++) {
				d_jobs[threadi].x[i] = 1.0;
			}
		}
		for (int i=0; i<d_jobs[threadi].n; i++) {
			d_jobs[threadi].x[i] = d_jobs[threadi].initial_conditions[i];
		}
	}
}

//------------------------------------------------------------------------------------------------------------------------------
__global__
void gpu_reset_dxdt(const int dev, const int numchips, job_struct_2* d_jobs) {
	for (int threadi = blockIdx.x * blockDim.x + threadIdx.x; threadi < numchips; threadi += blockDim.x * gridDim.x) {
		job_struct_2* job = &d_jobs[threadi];
		// reset dxdt:
		for (int i = 0; i < job->n + job->m*2; i++) {
			job->dxdt[i] = 0.0;
		}
		// reset loc:
		job->loc = job->m;
		// reset energy:
		job->energy = 0.0;
	}
}

//------------------------------------------------------------------------------------------------------------------------------
__global__
void gpu_step(const int dev, const int numchips, job_struct_2* d_jobs, state_struct* d_state) {
	for (int threadi = blockIdx.x * blockDim.x + threadIdx.x; threadi < numchips; threadi += blockDim.x * gridDim.x) {
		job_struct_2* job = &d_jobs[threadi];
		const PRECISION dmm_alpha_f 	= job->dmm_alpha;
		const PRECISION dmm_beta_f 		= job->dmm_beta;
		const PRECISION dmm_gamma_f 	= job->dmm_gamma;
		const PRECISION dmm_delta_f 	= job->dmm_delta;
		const PRECISION dmm_epsilon_f 	= job->dmm_epsilon;
		const PRECISION dmm_zeta_f 		= job->dmm_zeta;
		const uint32_t xl_max_f 		= job->xl_max;
		const int n_f 					= job->n;
		const int m_f 					= job->m;
		// loop through each clause:
		for (int clause = 0; clause < m_f; clause++) {
			const int a = job->cls[clause*MAX_LIT_SYSTEM+0];
			const int b = job->cls[clause*MAX_LIT_SYSTEM+1];
			const int c = job->cls[clause*MAX_LIT_SYSTEM+2];
			const int liti = abs(a);
			const int litj = abs(b);
			const int litk = abs(c);
			const PRECISION Qi = (a > 0)? 1.0:-1.0; // +1 if literal is >0, otherwise -1
			const PRECISION Qj = (b > 0)? 1.0:-1.0; // +1 if literal is >0, otherwise -1
			const PRECISION Qk = (c > 0)? 1.0:-1.0; // +1 if literal is >0, otherwise -1
			PRECISION Xs = job->x[clause+n_f]; if (Xs<0.0) Xs = 0.0; else if (Xs>1.0) Xs = 1.0; //Xs bounds
			PRECISION Xl = job->x[clause+n_f+m_f]; if (Xl<1.0) Xl = 1.0; else if (Xl>xl_max_f) Xl = PRECISION(xl_max_f); //Xl bounds
			// 3-sat:
			PRECISION Vi = job->x[liti-1]; if (Vi<-1.0) Vi = -1.0; else if (Vi>1.0) Vi = 1.0; //V bounds
			PRECISION Vj = job->x[litj-1]; if (Vj<-1.0) Vj = -1.0; else if (Vj>1.0) Vj = 1.0; //V bounds
			PRECISION Vk = job->x[litk-1]; if (Vk<-1.0) Vk = -1.0; else if (Vk>1.0) Vk = 1.0; //V bounds
			const PRECISION i = 1.0 - Qi*Vi;
			const PRECISION j = 1.0 - Qj*Vj;
			const PRECISION k = 1.0 - Qk*Vk;
			PRECISION C = fmin(i, fmin(j, k)) / 2.0; if (C < 0.0) C = 0.0; else if (C > 1.0) C = 1.0;
			//voltages:
			const PRECISION Gi = Qi * fmin(j, k) / 2.0;
			const PRECISION Gj = Qj * fmin(i, k) / 2.0;
			const PRECISION Gk = Qk * fmin(i, j) / 2.0;
			PRECISION Ri, Rj, Rk;
			if (C != i/2.0 ) {Ri = 0.0;} else {Ri = (Qi - Vi) / 2.0;}
			if (C != j/2.0 ) {Rj = 0.0;} else {Rj = (Qj - Vj) / 2.0;}
			if (C != k/2.0 ) {Rk = 0.0;} else {Rk = (Qk - Vk) / 2.0;}
			const PRECISION tmp1 = Xl * Xs;
			const PRECISION tmp2 = (1.0 + dmm_zeta_f * Xl) * (1.0 - Xs);
			job->dxdt[liti-1] += tmp1*Gi + tmp2 * Ri;
			job->dxdt[litj-1] += tmp1*Gj + tmp2 * Rj;
			job->dxdt[litk-1] += tmp1*Gk + tmp2 * Rk;

			// clause satsfied?
			if (C < 0.5) job->loc--;
			// update energy:
			job->energy += C;
			// Calculate Xs:
			job->dxdt[n_f + clause] = dmm_beta_f * (Xs + dmm_epsilon_f) * (C - dmm_gamma_f);
			// Calculate Xl:
			job->dxdt[n_f + m_f + clause] = dmm_alpha_f * (C - dmm_delta_f);
		}

		// only for debugging --------------------------------------------------------------------------------------------
		#ifdef GPUDEBUG
			if (job->loc < d_state[0].loc || job->energy < d_state[0].energy) {
			printf("CHIP %d: T=%f loc=%d energy=%f (global=%d) solved=%d alpha=%.5f stepsize=%.5f\n",threadi, job->t,  job->loc, job->energy, job->global, job->solved, job->dmm_alpha, job->stepsize);
			}
		#endif
		
		// ---------------------------------------------------------------------------------------------------------------

		// better global?
		if (job->loc < job->global) {
			job->global = job->loc;
			atomicMin(&d_state[0].loc, job->loc);
		}

		// globals?
		if (job->energy < d_state[0].energy) {
			d_state[0].energy = job->energy;
		}

		// solution found?
		if (job->loc == 0) {
			job->solved = true;
			//move to d_solution
			for (int i=0; i < job->n; i++) {
				d_state[0].solution[i] = (job->x[i] >= 0) ? true : false;
			}
		}

		// update steps:
		if (threadi == 0) d_state[0].steps++;

		// update time:
		job->t += job->stepsize;
	}
}

//------------------------------------------------------------------------------------------------------------------------------
__global__
void gpu_euler(const int dev, const int numchips, job_struct_2* d_jobs) {
	for (int threadi = blockIdx.x * blockDim.x + threadIdx.x; threadi < numchips; threadi += blockDim.x * gridDim.x) {
		job_struct_2* job = &d_jobs[threadi];
		const int n_f = job->n;
		const int m_f = job->m;
		const PRECISION stepsize_f = job->stepsize;
		const uint32_t xl_max_f = job->xl_max;

		for (int i = 0; i < n_f; i++) {
			// euler step:
			job->x[i] += stepsize_f * job->dxdt[i];
			// bounded variables:
			if (job->x[i] < -1.0) job->x[i] = -1.0; else if (job->x[i] > 1.0) job->x[i] =  1.0;
		}

		for (int i = n_f; i < n_f + m_f; i++) {
			// euler step:
			job->x[i] += stepsize_f * job->dxdt[i];
			// bounded variables:
			if (job->x[i] < 0.0) job->x[i] = 0.0; else if (job->x[i] > 1.0) job->x[i] = 1.0;
		}

		for (int i = n_f + m_f; i < n_f + m_f*2; i++) {
			// euler step:
			job->x[i] += stepsize_f * job->dxdt[i];
			// bounded variables:
			if (job->x[i] < 1.0) job->x[i] = 1.0; else if (job->x[i] > xl_max_f) job->x[i] = PRECISION(xl_max_f);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// new work, initiate state and load to GPU ---------------------------------------------------------------
int init_states_2(int device_id, int maximum_jobs, int use_random) {

	LogTS(TEXT_BCYAN) << "[INFO] INITIALIZING GPU STATES..." << std::endl;

	// base:
	uint64_cu mem_req = 0;
	mem_req += m * sizeof(int) * MAX_LIT_SYSTEM; //d_cls
	mem_req += sizeof(state_struct); //d_state
	mem_req += n * sizeof(bool); //d_solution
	mem_req += 1024; // reserved

	// per job:
	uint64_cu mem_job = 0;
	mem_job += sizeof(job_struct_2);
	mem_job += n * sizeof(PRECISION); //initial_conditions
	mem_job += (n + 2*m) * sizeof(PRECISION); //x
	mem_job += (n + 2*m) * sizeof(PRECISION); //dxdt
	mem_job += (3*m) * sizeof(int); //cls

	LogTS() << "[INFO] BASE MEMORY REQUIRED: " << mem_req << " BYTES" << std::endl;
	LogTS() << "[INFO] MIN MEMORY REQUIRED PER DYNEX CHIP: " << mem_job << " BYTES" << std::endl;

	LogTS() << "[INFO] SETTING MAX HEAP SIZES FOR GPUs..." << std::endl;
	// fitting jobs:
	int jobs_possible_all = 0;
	for (int dev = 0; dev < nDevices; dev++) {
		max_heap_size[dev] = 0;
		if (use_multi_gpu) device_id = dev;
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) != disabled_gpus.end()) continue;
		gpuErrchk(cudaSetDevice(device_id));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		uint64_cu mem_job_gpu = abs(mem_job*ADJ[dev]);
		size_t free, total; //tmp vars
		cudaMemGetInfo(&free, &total);
		size_t malloc_limit = free;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit);
		cudaDeviceGetLimit(&max_heap_size[dev], cudaLimitMallocHeapSize);
		LogTS() << "[GPU " << device_id << "] MAX HEAP: " << max_heap_size[dev] << " BYTES" << std::endl;
		if (max_heap_size[dev] < free) max_heap_size[dev] = free;
		if (max_heap_size[dev] <= mem_req + 2*mem_job_gpu) {
			LogTS() << "[GPU " << device_id << "] LOW MEMORY AVAILABLE - DISABLE" << std::endl;
			disabled_gpus.push_back(dev);
		} else {
			jobs_possible_all += (int)((max_heap_size[dev] - mem_req)/mem_job_gpu);
		}
	}

	LogTS() << "[INFO] MAX DYNEX CHIPS FITTING IN MEMORY (ALL GPUs): " << jobs_possible_all << std::endl;

	// num_jobs_all -> total jobs over all gpus:
	int num_jobs_all = jobs_possible_all;
	if (num_jobs_all > maximum_jobs) num_jobs_all = maximum_jobs; // user defined max #jobs

	if (!testing) {
		std::vector<std::string> p3;
		p3.push_back(MALLOB_NETWORK_ID);
		p3.push_back(std::to_string(JOB_ID));
		p3.push_back(std::to_string(num_jobs_all));
		p3.push_back(VERSION);
		jsonxx::Object o1 = mallob_mpi_command("cap", p3, 60);
		if (o1.get<jsonxx::Boolean>("updated")) {
			LogTS(TEXT_BGREEN) << "[MALLOB] UPDATING CAPACITY: SUCCESS" << std::endl;
		} else {
			LogTS(TEXT_BRED) << "[MALLOB] UPDATING CAPACITY: FAILED " << std::endl;
			return 0;
		}
	}

	LogTS(TEXT_BCYAN) << "[INFO] PREPARING " << num_jobs_all << " DYNEX CHIPS..." << std::endl;

	// loop through all GPUs:
	int num_jobs_free = num_jobs_all;

	for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev;
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) != disabled_gpus.end()) continue;

		gpuErrchk(cudaSetDevice(device_id));
		// calculate number of jobs to be created => num_jobs[dev]:
		int jobs_possible = 0;
		jobs_possible = (int)((max_heap_size[dev] - mem_req)/abs(mem_job*ADJ[dev]));
		// less jobs than space here?
		if (jobs_possible > num_jobs_free) {
			jobs_possible = num_jobs_free;
			nDevices = dev + 1; // reduce number of devices needed
		}
		num_jobs[dev] = jobs_possible;
		num_jobs_free = num_jobs_free - num_jobs[dev];

		LogTS() << "[GPU " << device_id << "] PREPARING " << num_jobs[dev] << " DYNEX CHIPS..." << std::endl;

		/// INIT MEMORY WITH KERNEL: ------------------------------------------------------------------------------------------
		LogTS() << "[GPU " << device_id << "] ALLOCATING MEMORY... " << std::endl;
		// create h_jobs and copy to d_jobs:
		int jobs_bytes = num_jobs[dev] * sizeof(job_struct_2);
		h_jobs_2[dev] = (job_struct_2*)calloc(num_jobs[dev], sizeof(job_struct_2));
		//copy jobs over to GPU (including sub arrays):
		uint64_cu mem_reserved = 0;
		for (int i = 0; i < num_jobs[dev]; i++) {
			gpuErrchk(cudaMalloc(&(h_jobs_2[dev][i].initial_conditions), (n) * sizeof(PRECISION)));
			gpuErrchk(cudaMalloc(&(h_jobs_2[dev][i].x), (n + 2*m) * sizeof(PRECISION)));
			gpuErrchk(cudaMalloc(&(h_jobs_2[dev][i].dxdt), (n + 2*m) * sizeof(PRECISION)));
			gpuErrchk(cudaMalloc(&(h_jobs_2[dev][i].cls), (3*m) * sizeof(int)));
			mem_reserved += (n + 2*m) * sizeof(PRECISION) * 2 + n * sizeof(PRECISION);
		}

		gpuErrchk(cudaMalloc((void**)&d_state[dev], sizeof(state_struct) + n*sizeof(bool)));
		gpuErrchk(cudaMalloc((void**)&d_jobs_2[dev], jobs_bytes)); //reserve memory for all jobs

		// cls, d_jobs:
		LogTS() << "[GPU " << device_id << "] COPYING PROBLEM... " << std::endl;
		gpuErrchk(cudaMemcpy(d_jobs_2[dev], h_jobs_2[dev], jobs_bytes, cudaMemcpyHostToDevice));
		// copy to thread 0
		gpuErrchk(cudaMemcpy(h_jobs_2[dev][0].cls, cls, m * MAX_LIT_SYSTEM * sizeof(int), cudaMemcpyHostToDevice));

		free(h_jobs_2[dev]);
		
		size_t free, total;
		cudaMemGetInfo(&free, &total);
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// set initial conditions:
	LogTS() << "[INFO] SETTING INITIAL CONDITIONS FOR ODE INTEGRATION AT T=0..." << std::endl;

	for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev;
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
			gpuErrchk(cudaSetDevice(device_id));
			gpu_set_initial_conditions <<< numBlocks[dev], threadsPerBlock[dev] >>> (dev, n, m, num_jobs[dev], d_jobs_2[dev], d_state[dev], init_dt, dmm_alpha, dmm_beta, dmm_gamma, dmm_delta, dmm_epsilon, dmm_zeta);
		}
	}

	for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev;
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
			gpuErrchk(cudaSetDevice(device_id));
			gpuErrchk(cudaDeviceSynchronize()); // wait for previous
			gpu_load_initial_conditions <<< numBlocks[dev], threadsPerBlock[dev] >>> (dev, num_jobs[dev], d_jobs_2[dev], use_random, testing);
		}
	}

	for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev;
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), dev) == disabled_gpus.end()) {
			gpuErrchk(cudaSetDevice(device_id));
			gpuErrchk(cudaDeviceSynchronize());
			// check init was really done
			state_struct h_state[1] = {0};
			gpuErrchk(cudaMemcpy(h_state, d_state[dev], sizeof(state_struct), cudaMemcpyDeviceToHost));
			if (h_state[0].loc == m && h_state[0].energy == (PRECISION)m) {
				LogTS(TEXT_SILVER) << "[GPU " << device_id << "] INITIALIZED" << std::endl;
			} else {
				LogTS(TEXT_BRED) << "[GPU " << device_id << "] INITIALIZATION FAILED" << std::endl;
				//return 0;
				num_jobs_all = 0; // exit later
			}
		}
	}
	return num_jobs_all;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
void gpu_reporting_runner(int device_id, int num_jobs_all) {
	auto t1 = std::chrono::steady_clock::now();
	uint32_t rem_unixts = 0, errors = 0;
	uint32_t invalid_timestamp_cnt = 0;

	std::this_thread::sleep_for(std::chrono::seconds(HASHRATE_INTERVAL));

	while (!dynex_quit_flag) {
		// screen output:
		std::string gpustats = "";
		int overall_loc_2 = overall_loc;
		int overall_energy_2 = overall_energy;
		float total_hashrate = 0.0;

		for (int dev = 0; dev < nDevices; dev++) {
			if (use_multi_gpu) device_id = dev;
			// only not disabled gpus:
			if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
				total_hashrate += overall_hashrates[dev];
				gpustats.append(gpustats == "" ? "[" : ",").append(std::to_string(overall_hashrates[dev]));
			}
		}
		if (gpustats != "") gpustats.append("]");

		auto t2 = std::chrono::steady_clock::now();
		float uptime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0;

		int hashrate_disp = (int)total_hashrate;
		LogTS(TEXT_SILVER) << "[GPU *] DYNEX CHIPS " << num_jobs_all << " | LOCAL MINIMA " << overall_loc_2 <<
			" | LOWEST ENERGY " << std::fixed << std::setprecision(2) << overall_energy_2 <<
			" | TOTAL HASHRATE " << hashrate_disp << " | UPTIME " << int(uptime) << std::endl;

		if (stratum) {
			uint64_t hashes = 0;
			uint32_t acc = 0, rej = 0, sta = 0, rej_series = 0;;
			dynexservice.getstats(&hashes, &acc, &rej, &sta, &rej_series);
			if (rej_series > MAX_REJECTED_SERIES) {
				LogTS(TEXT_BRED) << "[ERROR] MORE THAN " << MAX_REJECTED_SERIES << " REJECTED SHARES IN A ROW. QUITTING." << std::endl;
				dynex_quit_flag = true;
			}

			float hr = (uptime && hashes > 500) ? (hashes / uptime) : 0;
			LogTS(TEXT_BMAGENTA) << "[INFO] POOL HASHRATE " << int(hr) << " | ACCEPTED " << acc << " | REJECTED " << rej
				<< " | STALE " << sta << " | UPTIME " << int(uptime) << std::endl;
			if (STATS != "") {
				std::ofstream fout(STATS.c_str());
				fout << "{ \"ver\": \"" << VERSION << REVISION << "\", \"avg\": " << int(hr) << ", \"hr\": " << int(total_hashrate)
					<< ", \"ac\": " << acc << ", \"rj\": " << rej << ", \"st\": " << sta << ",\"gpu\": " << (gpustats==""?"null":gpustats)
					<< ", \"bus_numbers\": " << (BUSID==""?"null":BUSID) << ", \"uptime\": " << int(uptime) << " } " << std::endl;
				fout.close();
			}
		}

		if (!testing) {
			// atomic update:
			std::vector<std::string> p1;
			p1.push_back(MALLOB_NETWORK_ID);
			p1.push_back(std::to_string(1)); //status = running
			p1.push_back(std::to_string(overall_loc_2));
			p1.push_back(std::to_string(overall_energy_2));
			p1.push_back(std::to_string(total_hashrate));
			jsonxx::Object o1 = mallob_mpi_command("ato", p1, 60);
			if (o1.get<jsonxx::Boolean>("updated")) {
				errors = 0;
				LogTS() << "[MALLOB] ATOMIC STATUS UPDATED" << std::endl;
				if (atomic_updated == 0) atomic_updated = 1;
				// speedhack detection
				uint32_t unixts = std::strtoul(aes_decrypt(o1.get<jsonxx::String>("ts")).c_str(), NULL, 0);
				if (rem_unixts!=0) {
					uint32_t ts_diff = unixts - rem_unixts;
					if (ts_diff < 30 || ts_diff > 90) {
						LogTS(TEXT_BRED) << "[MALLOB] INVALID TIMESTAMP" << std::endl;
						invalid_timestamp_cnt++;
						if (invalid_timestamp_cnt>20) dynex_quit_flag = true;
					} else {
						invalid_timestamp_cnt = 0;
					}
				}
				rem_unixts = unixts;
			} else {
				errors++;
				LogTS(TEXT_BRED) << "[MALLOB] ATOMIC STATUS UPDATE: FAILED " << std::endl;
				rem_unixts = 0;
				if (errors > MAX_ATOMIC_ERR) dynex_quit_flag = true;
			}
		}
		auto t3 = std::chrono::steady_clock::now();
		int delay = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
		if (60000 - delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(60000 - delay));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
void gpu_runner(int device_id, int dev) {
	gpuErrchk(cudaSetDevice(device_id));
	int num_jobs_gpu = num_jobs[dev];

	if (debug) LogTS(TEXT_BCYAN) << "[GPU " << device_id << "] STARTING ODE INTEGRATION..." << std::endl;
	if (debug) LogTS() << "[GPU " << device_id << "] PARAMETERS: α=" << dmm_alpha << " β=" << dmm_beta << " γ=" << dmm_gamma
			<< " ε=" << dmm_delta << " δ=" << dmm_epsilon << " ζ=" << dmm_zeta << " initial d_t=" << init_dt << std::endl;

	state_struct* h_state = (state_struct*)calloc(sizeof(state_struct) + n*sizeof(bool), 1);

	uint64_cu prev_steps = 0;
	bool gpu_solved = false;
	int global_loc = m;
	PRECISION global_energy = (PRECISION)m;
	auto t1 = std::chrono::steady_clock::now();
	auto t2 = t1;
	float hashrate = 0.0;
	bool terminal = false;
	bool newminima = false;
	bool newenergy = false;

	// integration loop:
	while (!dynex_quit_flag) {

		// reset dxdt:
		gpu_reset_dxdt <<< numBlocks[dev], threadsPerBlock[dev] >>> (dev, num_jobs[dev], d_jobs_2[dev]);
		// gpu step:
		gpu_step <<< numBlocks[dev], threadsPerBlock[dev] >>> (dev, num_jobs[dev], d_jobs_2[dev], d_state[dev]);
		gpuErrchk(cudaMemcpy(h_state, d_state[dev], sizeof(state_struct), cudaMemcpyDeviceToHost));

		// update globals:
		if (h_state->loc < global_loc) {
			global_loc = h_state->loc; // update global loc in case
			newminima = true;
			atomic_fetch_min(&overall_loc, global_loc);
		}
		if (h_state->energy < global_energy) {
			global_energy = h_state->energy; // update global energy in case
			newenergy = true;
			atomic_fetch_min(&overall_energy, global_energy);
		}
		// solution found?
		if (h_state->loc == 0) {
			gpu_solved = true;
			terminal = true;
			gpuErrchk(cudaMemcpy(h_state, d_state[dev], sizeof(state_struct) + n*sizeof(bool), cudaMemcpyDeviceToHost));
		}

		if (atomic_updated == 1) dynexservice.update(num_jobs_gpu);

		// console?
		auto t3 = std::chrono::steady_clock::now();
		float passedtime = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()/1000.0;
		if (passedtime > HASHRATE_INTERVAL) terminal = true;

		// show status in terminal:
		if (terminal) {
			t2 = t3;
			float uptime = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()/1000.0;
			hashrate = (h_state->steps - prev_steps)*num_jobs_gpu/passedtime;
			overall_hashrates[dev] = factor*hashrate;
			prev_steps = h_state->steps;
			std::string loc_flag = (newminima) ? "*" : " ";
			std::string energy_flag = (newenergy) ? "*" : " ";
			int hashrate_disp = (int)overall_hashrates[dev];
			LogTS() << "[GPU " << device_id << "] DYNEX CHIPS " << num_jobs_gpu << " | STEPS " << h_state->steps << " | LOCAL MINIMA " << global_loc << loc_flag
				<< " | LOWEST ENERGY " << std::fixed << std::setprecision(2) << global_energy << energy_flag
				<< " | POUW HASHRATE " << hashrate_disp << " | UPTIME " << uptime << "s " << std::endl;
			terminal = false;
			newminima = false;
			newenergy = false;
		}

		if (gpu_solved) break;

		// apply rhs / ode integration step:
		gpu_euler <<< numBlocks[dev], threadsPerBlock[dev] >>> (dev, num_jobs[dev], d_jobs_2[dev]);
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// solution found?
	if (gpu_solved) {
		dynex_quit_flag = true; // quit all threads

		LogTS(TEXT_BYELLOW) << "[GPU " << device_id << "] SOLUTION FOUND!" << std::endl;
		// verify solution
		bool verify_sat = true;
		for (int j = 0; j < m; j++) {
			int lita = cls[j*MAX_LIT_SYSTEM+0]; bool a_pol = lita > 0 ? true : false;
			int litb = cls[j*MAX_LIT_SYSTEM+1]; bool b_pol = litb > 0 ? true : false;
			int litc = cls[j*MAX_LIT_SYSTEM+2]; bool c_pol = litc > 0 ? true : false;
			if (h_state->solution[abs(lita) - 1] != a_pol && h_state->solution[abs(litb) - 1] != b_pol && h_state->solution[abs(litc) - 1] != c_pol) {
				LogTS(TEXT_BRED) << "[ERROR] CLAUSE " << j << "[" << lita << " " << litb << " " << litc << "] HAS ASSIGNMENT "
					<< h_state->solution[abs(lita) - 1] << " " << h_state->solution[abs(litb) - 1] << " " << h_state->solution[abs(litc) - 1] << std::endl;
				verify_sat = false;
				break;
			}
		}
		if (!verify_sat) {
			LogTS(TEXT_BRED) << "[ERROR] SOLUTION NOT CERTIFIED" << std::endl;
		} else {
			LogTS(TEXT_BGREEN) << "[INFO] SOLUTION IS CERTIFIED" << std::endl;

			// output solution
			std::stringstream _solution;
			for (int i=0; i<n; i++) {
				_solution << (h_state->solution[i] ? i+1 : (i+1)*-1) << " ";
			}
			Log(TEXT_YELLOW) << "v " << _solution.str() << std::endl;

			//write solution to file:
			std::string solfile = JOB_FILENAME + ".solution.txt";
			FILE* fs = fopen(solfile.c_str(), "w");
			if (fs) {
				fprintf(fs, "%s\n", MINING_ADDRESS.c_str());
				for (int i=0; i<n; i++) {
					fprintf(fs, "%d, ", (h_state->solution[i] ? i+1 : (i+1)*-1));
				}
				fclose(fs);
				LogTS() << "[INFO] SOLUTION WRITTEN TO " << solfile << std::endl;
			}

			// submit solution to Dynex:
			if (!testing) {
				if (upload_file(solfile)) {
					LogTS(TEXT_BGREEN) << "[INFO] SOLUTION SUBMITTED TO DYNEX" << std::endl;
				}
			}

			if (!testing) {
				// atomic update:
				std::vector<std::string> p1;
				p1.push_back(MALLOB_NETWORK_ID);
				p1.push_back(std::to_string(2)); //status = solved
				p1.push_back(std::to_string(overall_loc));
				p1.push_back(std::to_string(overall_energy));
				p1.push_back(std::to_string(0));
				jsonxx::Object o1 = mallob_mpi_command("ato", p1, 60);
				if (o1.get<jsonxx::Boolean>("updated")) {
					LogTS(TEXT_SILVER) << "[MALLOB] ATOMIC STATUS UPDATED" << std::endl;
				} else {
					LogTS(TEXT_BRED) << "[MALLOB] ATOMIC STATUS NOT UPDATED " << std::endl;
				}
			}
		}
	}
	free(h_state);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// start dynexsolve
///////////////////////////////////////////////////////////////////////////////////////////////////////////
bool run_dynexsolve_2(int start_from_job, int maximum_jobs, int steps_per_batch, int device_id) {

	// initiate overall loc and energy:
	overall_loc = m;
	overall_energy = (PRECISION)m;
	for (int i=0; i < MAX_NUM_GPUS; i++) overall_hashrates[i] = 0;

	// configure threads and blocks:
	for (int i = 0; i < nDevices; i++) {
		numBlocks[i] = INTENSITY ? INTENSITY : 8192; 
		threadsPerBlock[i] = abs(num_jobs[i] / numBlocks[i]);
		if (numBlocks[i] < 1) numBlocks[i] = 1;
		if (threadsPerBlock[i] < 1) threadsPerBlock[i] = 1;
		if (debug) {
			LogTS() << "[DEBUG] GPU " << i << " threadsPerBlock = " << threadsPerBlock[i] << " numBlocks = " << numBlocks[i] << std::endl;
		}
	}

	int use_random = true;
	// init states for GPU:
	num_jobs_all = init_states_2(device_id, maximum_jobs, use_random);
	if (!num_jobs_all) return false;

	LogTS(TEXT_BCYAN) << "[INFO] STARTING ODE INTEGRATION..." << std::endl;

	// spawn a thread for each GPU:
	std::vector<std::thread> threads;
	for (int dev = 0; dev < nDevices; dev++) {
		if (use_multi_gpu) device_id = dev;
		// only not disabled gpus:
		if (std::find(disabled_gpus.begin(), disabled_gpus.end(), device_id) == disabled_gpus.end()) {
			threads.push_back(std::thread (gpu_runner, device_id, dev));
		}
	}

	if (!testing) {
		if (!dynexservice.start(threads.size(), STRATUM_URL, STRATUM_PORT, MINING_ADDRESS, STRATUM_PASSWORD, MALLOB_NETWORK_ID)) {
			LogTS(TEXT_BRED) << "[ERROR] CANNOT START DYNEX SERVICE" << std::endl;
			return false;
		}
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// reporting runner:
	std::thread rep_th(gpu_reporting_runner, device_id, num_jobs_all);
	rep_th.detach();

	for (auto &thread: threads) {
		thread.join();
	}

	// ---------------------------------------------------------------------------------------------------------------------
	// stop dynexservice:
	dynexservice.stop();

	// ---------------------------------------------------------------------------------------------------------------------
	if (!testing) {
		// atomic update:
		std::vector<std::string> p1;
		p1.push_back(MALLOB_NETWORK_ID);
		p1.push_back(std::to_string(3)); //status = cancelled
		p1.push_back(std::to_string(overall_loc));
		p1.push_back(std::to_string(overall_energy));
		p1.push_back(std::to_string(0));
		jsonxx::Object o1 = mallob_mpi_command("ato", p1, 60);
		if (o1.get<jsonxx::Boolean>("updated")) {
			LogTS(TEXT_SILVER) << "[MALLOB] ATOMIC STATUS UPDATED" << std::endl;
		} else {
			LogTS(TEXT_BRED) << "[MALLOB] ATOMIC STATUS UPDATE: FAILED "<< std::endl;
		}
	}

	// ---------------------------------------------------------------------------------------------------------------------

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
	LogTS() << " CTRL+C Interrupt signal (" << signum << ") received. Quitting gracefully..." << std::endl;

	// update mallob that we interruped:
	if (MALLOB_ACTIVE) {
		/// MALLOB: update_job_atomic -> let mallob know that we are working ++++++++++++++++++++++++++++++++++++++++++++++
		std::vector<std::string> p5;
		//network_id, atomic_status, steps_per_run, steps, hr, hr_adj
		p5.push_back(MALLOB_NETWORK_ID);
		p5.push_back(std::to_string(ATOMIC_STATUS_INTERRUPTED));
		p5.push_back(std::to_string(0));
		p5.push_back(std::to_string(0));
		p5.push_back(std::to_string(0));
		p5.push_back(std::to_string(0));
		jsonxx::Object o5 = mallob_mpi_command("update_atomic", p5, 60);
		if (o5.get<jsonxx::Boolean>("updated")) {
			LogTS(TEXT_SILVER) << "[INFO] MALLOB: ATOMIC JOB UPDATED" << std::endl;
		}
	/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}

	// stop miners:
	if (!SKIP) dynex_quit_flag = true; // stop signal to GPU job manager and CPU jobs
	if (!SKIP) dynexservice.dynex_hasher_quit_flag = true; // stop signal to Dynex hasher service
	if (!SKIP) LogTS(TEXT_SILVER) << "[INFO] FINISHING UP WORK ON GPU..." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	LogTS() << "[INFO] ---------------------------------------------------------" << std::endl;
	LogTS(TEXT_SILVER) << "[INFO] DynexSolve v" << VERSION << "(" << REVISION << ") | Meaningful Mining " << std::endl;
	LogTS() << "[INFO] ---------------------------------------------------------" << std::endl;

	// parse command line options:

	//help command?
	if (cmdOptionExists(argv, argv + argc, "-h"))
	{
		std::cout << "HELP" << std::endl;
		std::cout << "usage: dynexsolve -mining-address <WALLET ADDR> [options]" << std::endl;
		std::cout << std::endl;
		std::cout << "-mining-address <WALLET ADDR>    wallet address to receive the rewards" << std::endl;

		std::cout << "-stratum-url <HOST>              host of the stratum pool" << std::endl;
		std::cout << "-stratum-port <PORT>             port of the stratum pool" << std::endl;
		std::cout << "-stratum-paymentid <PAYMENT ID>  payment ID to add to wallet address" << std::endl;
		std::cout << "-stratum-password <PASSWORD>     stratum password (f.e. child@worker1)" << std::endl;
		std::cout << "-stratum-diff <DIFFICULTY>       stratum difficulty" << std::endl;

		std::cout << "-no-cpu                          run no Dynex chips on CPU" << std::endl;
		std::cout << "-no-gpu                          run no Dynex chips on GPU (WARNING: MINING NOT POSSIBLE)" << std::endl;
		std::cout << "-mallob-endpoint <IP>            set the endpoint for the Dynex Malleable Load Balancer" << std::endl;
		std::cout << "-devices                         show GPU devices on this system" << std::endl;
		std::cout << "-deviceid <GPU ID>               which GPU to use (default: 0 = first one) when using 1 GPU" << std::endl;
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
		std::cout << "-stats <FILENAME>                save stats in json format to file" << std::endl;
		std::cout << "-adj <DOUBLE,DOUBLE,DOUBLE>      adjust used mem amount per GPU (default: " << ADJ_DEFAULT << ")" << std::endl;
		std::cout << "-debug                           enable debugging output" << std::endl;
		std::cout << "-test <INPUTFILE>                test Dynex Chips locally" << std::endl;
		std::cout << "-h                               show help" << std::endl;
		return EXIT_SUCCESS;
	}

	//query devices?
	if (cmdOptionExists(argv, argv + argc, "-devices")) {
		int devnum = query_devices(-1);
		return EXIT_SUCCESS;
	}

	//test?
	char* tf = getCmdOption(argv, argv + argc, "-test");
	if (tf) {
		testing = true;
		testing_file = tf;
		MINING_ADDRESS = "XwnV1b9sULyFvmW8NGQyndJGWkF9eE13XKobuGvHUS4QFRrKH7Ze8tRFM6kPeLjLHyfLWPoo7r8RJKyqpcGxZHk32f2avgT4t";
		LogTS(TEXT_BGREEN) << "[INFO] TESTING ACTIVATED: " << testing_file << std::endl;
	}

	char* in = getCmdOption(argv, argv + argc, "-intensity");
	if (in) {
		INTENSITY = atoi(in);
		if (INTENSITY < 0) INTENSITY = 0;
		LogTS() << "[INFO] INTENSITY SET TO " << INTENSITY << std::endl;
	}

	//stratum
	char* surl = getCmdOption(argv, argv + argc, "-stratum-url");
	if (surl) {
		STRATUM_URL = surl;
		stratum = true;
		LogTS() << "[INFO] STRATUM PROTOCOL ENABLED " << std::endl;
		LogTS() << "[INFO] STRATUM URL SET TO " << STRATUM_URL << std::endl;
	}

	char* sport = getCmdOption(argv, argv + argc, "-stratum-port");
	if (sport) {
		STRATUM_PORT = atoi(sport);
		LogTS() << "[INFO] STRATUM PORT SET TO " << STRATUM_PORT << std::endl;
	}
	if (stratum && STRATUM_PORT <= 0) {
		LogTS(TEXT_BRED) << "[ERROR] INVALID PORT" << std::endl;
		return EXIT_FAILURE;
	}

	char* spay = getCmdOption(argv, argv + argc, "-stratum-paymentid");
	if (spay) {
		STRATUM_PAYMENT_ID = spay;
		LogTS() << "[INFO] STRATUM PAYMENT ID SET TO " << STRATUM_PAYMENT_ID << std::endl;
	}

	char* spass = getCmdOption(argv, argv + argc, "-stratum-password");
	if (spass) {
		STRATUM_PASSWORD = spass;
		LogTS() << "[INFO] STRATUM PASSWORD SET TO " << STRATUM_PASSWORD << std::endl;
	}

	char* sdiff = getCmdOption(argv, argv + argc, "-stratum-diff");
	if (sdiff) {
		STRATUM_DIFF = atoi(sdiff);
		LogTS() << "[INFO] STRATUM DIFF SET TO " << STRATUM_DIFF << std::endl;
	}

	//mining-address
	char* ma = getCmdOption(argv, argv + argc, "-mining-address");
	if (ma) {
		MINING_ADDRESS = ma + (stratum ? (STRATUM_PAYMENT_ID != "" ? "." + STRATUM_PAYMENT_ID : "") + (STRATUM_DIFF != 0 ? "." + std::to_string(STRATUM_DIFF) : "") : "");
		LogTS() << "[INFO] MINING ADDRESS SET TO " << MINING_ADDRESS << std::endl;
	}

	if (MINING_ADDRESS=="") {
		LogTS(TEXT_BRED) << "[ERROR] WALLET ADDRESS NOT SPECIFIED" << std::endl;
		return EXIT_FAILURE;
	}

	//mallob endpoint?
	char* me = getCmdOption(argv, argv + argc, "-mallob-endpoint");
	if (me) {
		mallob_endpoint = me;
		LogTS() << "[INFO] OPTION mallob-endpoint SET TO " << mallob_endpoint << std::endl;
	}

	//debugger?
	bool dynex_debugger = false;
	if (cmdOptionExists(argv, argv + argc, "-debug")) {
		dynex_debugger = true;
		debug = dynex_debugger;
		LogTS() << "[INFO] OPTION debug ACTIVATED" << std::endl;
	}

	//- multi - gpu
	if (cmdOptionExists(argv, argv + argc, "-multi-gpu")) {
		use_multi_gpu = true;
		LogTS() << "[INFO] OPTION multi-gpu ACTIVATED" << std::endl;
	}

	//disable gpu?
	bool disable_gpu = false;
	if (cmdOptionExists(argv, argv + argc, "-no-gpu")) {
		disable_gpu = true;
		LogTS() << "[INFO] OPTION no-gpu ACTIVATED - "; Log(TEXT_BRED) << "ONLY SEARCHING FOR SOLUTION REWARD" << std::endl;
	}

	//disable certain?
	char* dgp = getCmdOption(argv, argv + argc, "-disable-gpu");
	if (dgp) {
		if (use_multi_gpu) {
			std::string disable_gpus = dgp;
			std::vector<std::string>disabled_gpus_str = split(disable_gpus,',');
			for (int i=0; i<disabled_gpus_str.size(); i++) disabled_gpus.push_back(atoi(disabled_gpus_str[i].c_str()));
			LogTS() << "[INFO] OPTION disable-gpu SET TO " << disable_gpus << std::endl;
		} else  {
			LogTS(TEXT_BRED) << "[ERROR] Option -disable-gpu cannot be used without option -multi-gpu" << std::endl;
			return EXIT_FAILURE;
		}
	}

	//alpha, beta, gamma, delta, epsilon, zeta:
	
	char* a = getCmdOption(argv, argv + argc, "-alpha");
	if (a) {
		dmm_alpha = atof(a);
		LogTS() << "[INFO] OPTION alpha SET TO " << dmm_alpha << std::endl;
	}
	char* b = getCmdOption(argv, argv + argc, "-beta");
	if (b) {
		dmm_beta = atof(b);
		LogTS() << "[INFO] OPTION beta SET TO " << dmm_beta << std::endl;
	}
	char* g = getCmdOption(argv, argv + argc, "-gamma");
	if (g) {
		dmm_gamma = atof(g);
		LogTS() << "[INFO] OPTION gamma SET TO " << dmm_gamma << std::endl;
	}
	char* d = getCmdOption(argv, argv + argc, "-delta");
	if (d) {
		dmm_delta = atof(d);
		LogTS() << "[INFO] OPTION delta SET TO " << dmm_delta << std::endl;
	}
	char* e = getCmdOption(argv, argv + argc, "-epsilon");
	if (e) {
		dmm_epsilon = atof(e);
		LogTS() << "[INFO] OPTION epsilon SET TO " << dmm_epsilon << std::endl;
	}
	char* z = getCmdOption(argv, argv + argc, "-zeta");
	if (z) {
		dmm_zeta = atof(z);
		LogTS() << "[INFO] OPTION zeta SET TO " << dmm_zeta << std::endl;
	}
	char* dt = getCmdOption(argv, argv + argc, "-init_dt");
	if (dt) {
		init_dt = atof(dt);
		LogTS() << "[INFO] OPTION init_dt SET TO " << init_dt << std::endl;
	}

	std::vector<std::string>adj_gpu;
	char* da = getCmdOption(argv, argv + argc, "-adj");
	if (da) {
		adj_gpu = split(da,',');
	}
	float adj_last = ADJ_DEFAULT;
	for (int i=0; i < MAX_NUM_GPUS; i++) {
		float adj = (i < adj_gpu.size()) ? atof(adj_gpu[i].c_str()) : adj_last;
		if (adj < 0.8) adj = 0.8;
		ADJ[i] = adj;
		adj_last = adj;
	}

	//cpu_chips?
	int cpu_chips = 0;
	char* rc = getCmdOption(argv, argv + argc, "-cpu-chips");
	if (rc) {
		cpu_chips = atoi(rc);
		if (cpu_chips < 0) cpu_chips = 0;
		if (cpu_chips > std::thread::hardware_concurrency()) cpu_chips = std::thread::hardware_concurrency();
		LogTS() << "[INFO] OPTION cpu-chips SET TO " << cpu_chips << std::endl;
	}

	//disable cpu?
	if (cmdOptionExists(argv, argv + argc, "-no-cpu")) {
		cpu_chips = 0;
		LogTS() << "[INFO] OPTION no-cpu ACTIVATED" << std::endl;
	}

	//start_from_job specified?
	int start_from_job = 0;
	char* sfj = getCmdOption(argv, argv + argc, "-start-from-job");
	if (sfj) {
		start_from_job = atoi(sfj);
		LogTS() << "[INFO] OPTION start-from-job SET TO " << start_from_job << std::endl;
	}

	//maximum_chips specified?
	int maximum_jobs = INT_MAX;
	char* mj = getCmdOption(argv, argv + argc, "-maximum-chips");
	if (mj) {
		maximum_jobs = atoi(mj);
		LogTS() << "[INFO] OPTION maximum-chips SET TO " << maximum_jobs << std::endl;
	}

	//maximum_jobs specified?
	int steps_per_batch = 10000;
	char* spb = getCmdOption(argv, argv + argc, "-steps-per-batch");
	if (spb) {
		steps_per_batch = atoi(spb);
		if (steps_per_batch < 10000) steps_per_batch = 10000;
		LogTS() << "[INFO] OPTION steps-per-batch SET TO " << steps_per_batch << std::endl;
	}

	//deviceid specified?
	int device_id = 0;
	char* did = getCmdOption(argv, argv + argc, "-deviceid");
	if (did) {
		device_id = atoi(did);
		LogTS() << "[INFO] OPTION deviceid SET TO " << device_id << std::endl;
		use_multi_gpu = false;
	}

	char* st = getCmdOption(argv, argv + argc, "-stats");
	if (st) {
		std::ofstream fout(st);
		if (fout.is_open()) {
			STATS = st;
			LogTS() << "[INFO] OPTION stats SET TO " << STATS << std::endl;
			fout << "{ \"ver\": \"" << VERSION << REVISION << "\", \"hr\": " << 0 << ", \"ac\": " << 0 << ", \"rj\": " << 0 << ", \"uptime\": " << 0 << " } " << std::endl;
			fout.close();
		} else {
			LogTS(TEXT_BRED) << "[ERROR] Unable to create stats file: " << STATS << std::endl;
		}
	}
	// ------------------------------------ end command line parameters --------------------------------------------------------------------

	// single or multi gpu?:
	if (!disable_gpu) {
		cudaGetDeviceCount(&nDevices);
		if (!use_multi_gpu) {
			nDevices = 1;
		} else {
			// multi gpu:
			LogTS(TEXT_SILVER) << "[INFO] MULTI-GPU ENABLED" << std::endl;
			device_id = -1;
		}
		query_devices(device_id);
	}

	curl_global_init(CURL_GLOBAL_DEFAULT);

	/// MALLOB ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if (MALLOB_NETWORK_ID == "") {
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_int_distribution<uint64_t> dis;
		std::stringstream sstream;
		for (int i = 0; i < 4; i++) {
			sstream << std::setw(16) << std::setfill('0') << std::hex << dis(gen);
		}
		MALLOB_NETWORK_ID = sstream.str();
		srand(dis(gen));
	}

	std::string SHA3HASH{}; 

	// Register as new worker:
	if (!testing) {
		std::vector<std::string> p1;
		p1.push_back(VERSION);
		p1.push_back(MALLOB_NETWORK_ID);
		p1.push_back(MINING_ADDRESS+(STRATUM_PASSWORD!="" ? ":"+STRATUM_PASSWORD : ""));
		jsonxx::Object o1 = mallob_mpi_command("reg", p1, 60);
		if (o1.get<jsonxx::Boolean>("registered")) {
			LogTS(TEXT_BGREEN) << "[MALLOB] REGISTER WORKER: SUCCESS" << std::endl;
			MALLOB_ACTIVE = true;
		} else {
			LogTS(TEXT_BRED) << "[MALLOB] REGISTER WORKER: FAILED" << std::endl;
			return EXIT_FAILURE;
		}
		int CHIPS_AVAILABLE, CHIPS_REQUIRED;
		double JOB_FEE, JOB_SOLUTION_REWARD;
		if (o1.has<jsonxx::String>("data")) {
			jsonxx::Object data;
			if (!data.parse(aes_decrypt(o1.get<jsonxx::String>("data")))) {
				LogTS(TEXT_BRED) << "[MALLOB] REGISTER WORKER: INVALID DATA" << std::endl;
				return EXIT_FAILURE;
			}
			//if (debug) LogTS() << data.json();
			JOB_ID = data.get<jsonxx::Number>("id");
			CHIPS_AVAILABLE = data.get<jsonxx::Number>("chips_available");
			CHIPS_REQUIRED = data.get<jsonxx::Number>("chips_required");
			JOB_FILENAME = data.get<jsonxx::String>("filename");
			JOB_FEE = data.get<jsonxx::Number>("fee");
			JOB_SOLUTION_REWARD = data.get<jsonxx::Number>("reward");
			dmm_alpha = data.get<jsonxx::Number>("P1");
			dmm_beta = data.get<jsonxx::Number>("P2");
			dmm_gamma = data.get<jsonxx::Number>("P3");
			dmm_delta = data.get<jsonxx::Number>("P4");
			dmm_epsilon = data.get<jsonxx::Number>("P5");
			dmm_zeta = data.get<jsonxx::Number>("P6");
			init_dt = data.get<jsonxx::Number>("P7");
			CNF_DOWNLOADURL = data.get<jsonxx::String>("downloadurl");
			CNF_SOLUTIONURL = data.get<jsonxx::String>("solutionurl");
			CNF_SOLUTIONUSER = data.get<jsonxx::String>("solutionuser");
			factor = data.get<jsonxx::Number>("factor");
			if (data.has<jsonxx::String>("network_id")) {
				MALLOB_NETWORK_ID = data.get<jsonxx::String>("network_id");
			}
			SHA3HASH = data.get<jsonxx::String>("sha3");
		} else {
			LogTS(TEXT_BRED) << "[MALLOB] REGISTER WORKER: INVALID DATA" << std::endl;
			return EXIT_FAILURE;
		}

		LogTS(TEXT_SILVER) << "[MALLOB] JOB RECEIVED        : " << JOB_ID << std::endl;
		LogTS(TEXT_SILVER) << "[MALLOB] CHIPS AVAILABLE     : " << CHIPS_AVAILABLE << "/" << CHIPS_REQUIRED << std::endl;
		LogTS(TEXT_SILVER) << "[MALLOB] JOB FILENAME        : " << JOB_FILENAME << std::endl;
		LogTS(TEXT_SILVER) << "[MALLOB] JOB FEE             : BLOCK REWARD + " << JOB_FEE <<  " DNX" << std::endl;
		LogTS(TEXT_SILVER) << "[MALLOB] JOB SOLUTION REWARD : " << JOB_SOLUTION_REWARD <<  " DNX" << std::endl;
		LogTS(TEXT_SILVER) << "[MALLOB] PARAMETERS: α=" << dmm_alpha << " β=" << dmm_beta << " γ=" << dmm_gamma << " ε=" << dmm_delta
			<< " δ=" << dmm_epsilon << " ζ=" << dmm_zeta << " initial d_t=" << init_dt << std::endl;

		// double check; chips also available?
		if (CHIPS_AVAILABLE <= 0) {
			LogTS(TEXT_BRED) << "[MALLOB] NO JOBS AVAILABLE" << std::endl;
			return EXIT_FAILURE;
		}
	}

	LogTS() << "[MALLOB] NETWORK ID " << MALLOB_NETWORK_ID << std::endl;

	// sanity check: mallob_network_id 64 bytes?
	if (MALLOB_NETWORK_ID.size() != 64) {
		LogTS(TEXT_BRED) << "[ERROR] NETWORK ID HAS THE WRONG SIZE. ABORT" << std::endl;
		return EXIT_FAILURE;
	}

	//convert(MALLOB_NETWORK_ID.c_str(), network_id, sizeof(network_id));

	// testing?
	if (testing) JOB_FILENAME = testing_file;

	// file existing?
	if (!file_exists(JOB_FILENAME)) {
		LogTS() << "[MALLOB] DOWNLOADING JOB: " << JOB_FILENAME << std::endl;
		if (!download_file(JOB_FILENAME)) return EXIT_FAILURE;
		LogTS(TEXT_BGREEN) << "[MALLOB] JOB SUCCESSFULLY DOWNLOADED" << std::endl;
	}

	/// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	if (!load_cnf(JOB_FILENAME, SHA3HASH)) {
		std::remove(JOB_FILENAME.c_str()); // delete broken file
		return EXIT_FAILURE;
	}
	LogTS() << "[INFO] FORMULATION LOADED" << std::endl;

	// run CPU dynex chips
	bool dnxret = dynexchip.start(cpu_chips, JOB_FILENAME, std::ref(dynex_quit_flag), dmm_alpha, dmm_beta, dmm_gamma, dmm_delta, dmm_epsilon, dmm_zeta, init_dt, dynex_debugger, steps_per_batch);

	if (disable_gpu) {
		while (!dynex_quit_flag) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		return EXIT_SUCCESS;
	}

	// run GPU dynex chips:
	if (!run_dynexsolve_2(start_from_job, maximum_jobs, steps_per_batch, device_id)) {
		LogTS(TEXT_BRED) << "[ERROR] EXIT WITH ERROR" << std::endl;
		return EXIT_FAILURE;
	}

	auto t2 = std::chrono::steady_clock::now();
	LogTS() << "[INFO] WALL TIME: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count()
		<< "ms" << std::endl;

	curl_global_cleanup();

	LogTS() << "GOOD BYE!" << std::endl;

	return EXIT_SUCCESS;
}
