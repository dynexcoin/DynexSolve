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

#pragma comment(lib, "libcurl.lib" )
#pragma comment(lib, "winmm.lib" )
#pragma comment(lib, "ws2_32.lib")
#pragma comment (lib, "zlib.lib")
#pragma comment (lib, "advapi32.lib")
#pragma comment (lib, "crypt32.lib")

#include <future>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef WIN32
#include <pthread.h>
#endif // !WIN32
#include <signal.h>
#include <math.h>
#include <stdbool.h>
#include <locale.h>
#include <random>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#ifdef WIN32
	#pragma comment(lib,"Crypto.lib")
#endif // WIN32
extern "C"
void cn_slow_hash(const void* data, size_t length, uint8_t * hash);

//#define CURL_STATICLIB
#include <curl/curl.h> //required for MPI - dependency
#include "jsonxx.h"    //no install required + cross platform - https://github.com/hjiang/jsonxx
// stratum:
#include "tcp_sockets.hpp" // - https://stlplus.sourceforge.net/stlplus3/docs/tcp_sockets.html

static bool stratum_connected = false;
static std::string H_STRATUM_JOB_ID = "";
static std::string H_STRATUM_ID = "";
static std::string H_STRATUM_BLOB = "";
static std::string H_STRATUM_WALLET = "";
static std::string H_STRATUM_TARGET = "";
static bool H_stratum = false;
static std::string H_STRATUM_URL = "";
static std::string H_STRATUM_PORT = "";
static std::string H_STRATUM_PAYMENT_ID = "";
static std::string H_STRATUM_PASSWORD = "";
static int         H_STRATUM_DIFF = 0;
static std::string H_POW = "";
static stlplus::TCP_client stratumclient;
static int accepted_cnt = 0;
static int rejected_cnt = 0;
static std::string H_MALLOB_NETWORK_ID = "";

// Dynex colors
#ifdef WIN32
#define TEXT_DEFAULT  ""
#define TEXT_YELLOW   ""
#define TEXT_GREEN    ""
#define TEXT_RED      ""
#define TEXT_BLUE     ""
#define TEXT_CYAN     ""
#define TEXT_WHITE    ""
#define TEXT_SILVER   ""
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


// block_template: ----------------------------------------------------------------------------------------
typedef struct {
	std::string blockhashing_blob;
	std::string blocktemplate_blob;
	uint64_t difficulty;
	uint64_t height;
	int reserved_offset;
	uint32_t nonce;
} block_template;

typedef struct {
	int depth;
	uint64_t difficulty;
	std::string hash;
	uint64_t height;
	int major_version;
	int minor_version;
	uint32_t nonce;
	bool orphan_status;
	std::string prev_hash;
	uint64_t reward;
	uint64_t timestamp;
} block_header;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// HELPER FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convert 8 x uint8_t to uint64_t hex
static inline uint64_t bytesToInt64(uint8_t bytes[8])
{
   uint64_t v = 0;
   v |= bytes[0]; v <<= 8;
   v |= bytes[1]; v <<= 8;
   v |= bytes[2]; v <<= 8;
   v |= bytes[3]; v <<= 8;
   v |= bytes[4]; v <<= 8;
   v |= bytes[5]; v <<= 8;
   v |= bytes[6]; v <<= 8;
   v |= bytes[7];
   return v;
}

// convert string to uint8_t:
static inline void convert(const char *s, int size, uint8_t * out) {
  int i = 0;
  while (*s) {
    char byte[3] = { *s, *(s + 1), 0 };
    out[i++] = strtol(byte, NULL, 16);
    s += 2;
  }
  for (; i < 8; i += 1) {
    out[i] = 0;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bit functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline uint64_t swap64(uint64_t x) {
  x = ((x & 0x00ff00ff00ff00ff) <<  8) | ((x & 0xff00ff00ff00ff00) >>  8);
  x = ((x & 0x0000ffff0000ffff) << 16) | ((x & 0xffff0000ffff0000) >> 16);
  return (x << 32) | (x >> 32);
}
#define swap64le swap64 //ident64 
static inline uint64_t hi_dword(uint64_t val) {return val >> 32;}
static inline uint64_t lo_dword(uint64_t val) {return val & 0xFFFFFFFF;}
static inline uint64_t mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi) {
  uint64_t a = hi_dword(multiplier);
  uint64_t b = lo_dword(multiplier);
  uint64_t c = hi_dword(multiplicand);
  uint64_t d = lo_dword(multiplicand);
  uint64_t ac = a * c;
  uint64_t ad = a * d;
  uint64_t bc = b * c;
  uint64_t bd = b * d;
  uint64_t adbc = ad + bc;
  uint64_t adbc_carry = adbc < ad ? 1 : 0;
  uint64_t product_lo = bd + (adbc << 32);
  uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
  *product_hi = ac + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;
  assert(ac <= *product_hi);
  return product_lo;
}
static inline void mul(uint64_t a, uint64_t b, uint64_t &low, uint64_t &high) {low = mul128(a, b, &high);}
static inline uint64_t ident64(uint64_t x) { return x; }
static inline bool cadd(uint64_t a, uint64_t b) {return a + b < a;}
static inline bool cadc(uint64_t a, uint64_t b, bool c) {return a + b < a || (c && a + b == (uint64_t) -1);}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynex Service Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class dynex_hasher_thread_obj {

	std::promise<void> exitSignal;
	std::future<void> futureObj;
	
	public:
		uint64_t hashcounter = 0;
		
		// operator function --------------------------------------------------------------------------------------------
		void operator()(int thread_id,int& sleep_time_ms,  uint64_t& leffom, std::atomic<bool>& hasher_quit_flag, std::string daemon_host, std::string daemon_port, CURL* curl, std::string address, int reserve_size) 
		{
			bool workfinished = hasher_work(thread_id, sleep_time_ms, leffom, hasher_quit_flag, daemon_host, daemon_port, curl, address, reserve_size);
			return;
		}
		// ---------------------------------------------------------------------------------------------------------------
		
	private:
		// pretty log time: 
		std::string log_time() {
		    auto t = std::time(nullptr);
		    auto tm = *std::localtime(&t);

		    std::ostringstream oss;
		    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
		    auto str = oss.str();

		    return str;
		}

		// curl return value function:
		static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp){
		    ((std::string*)userp)->append((char*)contents, size * nmemb);
		    return size * nmemb;
		}

		// check difficulty
		bool check_hash(const uint64_t* hash, const uint64_t difficulty) {
			uint64_t low, high, top, cur;
			mul(swap64le(hash[3]), difficulty, top, high);
			if (high != 0) { return false; }
			mul(swap64le(hash[0]), difficulty, low, cur);
			mul(swap64le(hash[1]), difficulty, low, high);
			bool carry = cadd(cur, low);
			cur = high;
			mul(swap64le(hash[2]), difficulty, low, high);
			carry = cadc(cur, low, carry);
			carry = cadc(high, top, carry);
			return !carry;
		}

		// try_hash: ----------------------------------------------------------------------------------------
		bool try_hash(std::string blockhashing_blob, uint64_t nonce, uint64_t difficulty, bool logging = false) {
			if (difficulty == 0) return false;
			// convert nonce to hex:
			std::stringstream ss;
		    	ss << std::hex << std::setfill('0') << std::setw(sizeof(uint64_t)) << static_cast<int>(nonce);
		    	std::string nonce_hex = ss.str().substr(6,2) + ss.str().substr(4,2) + ss.str().substr(2,2) + ss.str().substr(0,2); //little endian
			if (logging) std::cout << "nonce_hex: " << nonce_hex << std::endl; 
			// add nonce into blob => input
			std::string input = blockhashing_blob;
			input.replace(78, 8, nonce_hex); 
			if (logging) std::cout << "INPUT: " << input << std::endl;
			// convert input to unint8_t => input_hex
			uint8_t input_size = input.size()/2;
			uint8_t * input_hex = new uint8_t[input_size];
			convert(input.c_str(), input.size(), input_hex);
			if (logging) {
				std::cout << "INPUT (HEX uint8_t)    :";
				for (int i = 0; i < input_size; i++) printf("%02x", input_hex[i]);
				printf("\n");	
			}
			uint8_t hash[32];
			cn_slow_hash(input_hex, input_size, hash); 
			// convert output hash to std::string:
			std::stringstream _outhash; _outhash << std::hex << std::setfill('0');
			for (int i = 0; i < 32; i++) _outhash << std::hex << std::setw(2) << static_cast<int>((uint8_t)hash[i]);
			std::string outhash = _outhash.str();
			if (logging) std::cout << "OUTPUT (std::string) : " << outhash << std::endl;
			if (H_stratum) H_POW = outhash; // stratum needs proof of work
			// hash -> hash_64 (uint64): 
			uint64_t hash64[4]; int hashpos = 0;
			#pragma unroll
			for (int i=0; i< 32; i = i + 8) {
				uint8_t bytes[8]; 
				#pragma unroll 
				for (int ii=0; ii<8; ii++) bytes[ii] = hash[i+ii];
				hash64[hashpos] = bytesToInt64(bytes);
				hashpos++;
			}
			if (logging) {
				std::cout << "OUTPUT (HEX uint64_t): ";
				for (int i=0; i< 4; i++) std::cout << std::hex << std::setw(16) << std::setfill('0') << hash64[i];		
				std::cout << std::endl;
			}
			// nonce found?
			bool foundnonce = check_hash(hash64, difficulty);
			//if (foundnonce) std::cout << " debug: check_hash " << outhash << " with difficulty " << difficulty << " result: TRUE" << std::endl;
			if (foundnonce) return true;
			return false;
		}
		
		// rpc command handler: -------------------------------------------------------------------------------------------------
		jsonxx::Object invokeJsonRpcCommand(std::string method, std::vector<std::string> params, std::string daemon_host, std::string daemon_port, CURL* curl) {
			jsonxx::Object retval;
			
			std::string url = "http://" + daemon_host + ":" + daemon_port + "/json_rpc";
			std::string postfields = "{\"jsonrpc\":\"2.0\",\"method\":\""+method+"\",\"params\":{";
			
			if (params.size()>0) {
				for (int i=0; i<params.size(); i++) postfields = postfields+params[i]+",";
				postfields.pop_back();
			}
			postfields = postfields + "}}";
			//std::cout << "postfields: " << postfields << std::endl;
			//CURL *curl;
			CURLcode res;
			std::string readBuffer;
			//curl_global_init(CURL_GLOBAL_ALL);
			curl = curl_easy_init();
			if(curl) {
				curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
				curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postfields.c_str());
				curl_easy_setopt(curl, CURLOPT_VERBOSE, 0);
				curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L); // 5s
				curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L); // 5 s
				curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
				curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
				res = curl_easy_perform(curl);
				curl_easy_cleanup(curl);
				if(res != CURLE_OK) {
					fprintf(stderr, "Connection to daemon failed: %s\n", curl_easy_strerror(res));
					retval.parse("{\"ERROR\":true}");
				} else {
					//std::cout << "readBuffer: " << readBuffer << std::endl;
					std::istringstream input(readBuffer);
					retval.parse(input);
				}
				
			}
			//curl_global_cleanup();
			return retval;
		}
		
		// get_block_template ------------------------------------------------------------------------------------------------------
		block_template get_blocktemplate(std::string address, int reserve_size, std::string daemon_host, std::string daemon_port, CURL* curl) {
			block_template bt;
			bt.difficulty = 0;
			std::vector<std::string> params;
			params.push_back("\"reserve_size\":"+std::to_string(reserve_size)); 
			params.push_back("\"wallet_address\":\""+address+"\"");
			jsonxx::Object retval = invokeJsonRpcCommand("getblocktemplate",params, daemon_host, daemon_port, curl);
			if (!retval.has<jsonxx::Boolean>("ERROR")) {
				//std::cout << TEXT_GREEN << "parsed " << retval.json() << TEXT_DEFAULT << std::endl;
				jsonxx::Object result = retval.get<jsonxx::Object>("result");
				//std::cout << TEXT_GREEN << "parsed " << result.json() << TEXT_DEFAULT << std::endl;
				bt.blockhashing_blob = result.get<jsonxx::String>("blockhashing_blob");
				bt.blocktemplate_blob = result.get<jsonxx::String>("blocktemplate_blob");
				bt.difficulty = result.get<jsonxx::Number>("difficulty");
				bt.height = result.get<jsonxx::Number>("height");
				bt.reserved_offset = result.get<jsonxx::Number>("reserved_offset");
				bt.nonce = 0;
			}
			return bt;
		}
		
		// getlastblockheader -------------------------------------------------------------------------------------
		block_header getlastblockheader(std::string daemon_host, std::string daemon_port, CURL* curl) {
			block_header bh;
			bh.difficulty = 0;
			std::vector<std::string> params;
			jsonxx::Object retval = invokeJsonRpcCommand("getlastblockheader",params, daemon_host, daemon_port, curl);
			if (!retval.has<jsonxx::Boolean>("ERROR")) {
				//std::cout << TEXT_GREEN << "parsed " << retval.json() << TEXT_DEFAULT << std::endl;
				jsonxx::Object result = retval.get<jsonxx::Object>("result");
				jsonxx::Object blockheader = result.get<jsonxx::Object>("block_header");
				//std::cout << TEXT_GREEN << "parsed " << blockheader.json() << TEXT_DEFAULT << std::endl;
				bh.depth = blockheader.get<jsonxx::Number>("depth"); 
				bh.difficulty = blockheader.get<jsonxx::Number>("difficulty");
				bh.hash = blockheader.get<jsonxx::String>("hash");
				bh.height = blockheader.get<jsonxx::Number>("height");
				bh.major_version = blockheader.get<jsonxx::Number>("major_version");
				bh.minor_version = blockheader.get<jsonxx::Number>("minor_version");
				bh.nonce = blockheader.get<jsonxx::Number>("nonce");
				bh.orphan_status = blockheader.get<jsonxx::Boolean>("orphan_status");
				bh.prev_hash = blockheader.get<jsonxx::String>("prev_hash");
				bh.reward = blockheader.get<jsonxx::Number>("reward");
				bh.timestamp = blockheader.get<jsonxx::Number>("timestamp");
			}
			return bh;
		}
		
		// submit block ----------------------------------------------------------------------------------------------
		bool submitblock(std::string blockdata, std::string daemon_host, std::string daemon_port, CURL* curl) {
			std::vector<std::string> params;
			jsonxx::Object retval;
			std::string url = "http://" + daemon_host + ":" + daemon_port + "/json_rpc";
			std::string postfields = "{\"jsonrpc\":\"2.0\",\"method\":\"submitblock\",\"params\":[\""+blockdata+"\"],\"mallob\":\""+H_MALLOB_NETWORK_ID+"\"}";
			CURLcode res;
			std::string readBuffer;
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
					fprintf(stderr, "submitblock failed: %s\n", curl_easy_strerror(res));
				} else {
					//std::cout << "readBuffer: " << readBuffer << std::endl;
					std::istringstream input(readBuffer);
					if (input.str() == "{\"jsonrpc\":\"2.0\",\"result\":{\"status\":\"OK\"}}") {
						return true;
					}
				}
				
			}
			
			return false;
		}
		
		// Stratum: login ----------------------------------------------------------------------------------------------
		bool stratum_login() {
			std::cout << log_time() << " [STRATUM] CONNECTING TO " << H_STRATUM_URL << ":" << H_STRATUM_PORT << std::endl;
			
			// reconnect properly:
			stratumclient.close();
			stlplus::TCP_client client(H_STRATUM_URL.c_str(),(unsigned short)atoi(H_STRATUM_PORT.c_str()), 10000000);
			stratumclient = client;
			//stratumclient.initialise(H_STRATUM_URL.c_str(),(unsigned short)atoi(H_STRATUM_PORT.c_str()),5000000);
			
			if (!stratumclient.initialised())
			    {
			      std::cout << log_time() << " [STRATUM] client failed to initialise" << std::endl;
			      return false;
			    }
			    if (stratumclient.error())
			    {
			      std::cout << log_time() << " [STRATUM] client initialisation failed with error " << stratumclient.error() << std::endl;
			      return false;
			    }
			
			std::cout << log_time() << " [STRATUM] CONNECTED, LOGGING IN... " << std::endl;

			std::string COMMAND_LOGIN;
			if (H_STRATUM_DIFF==0) {
					COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\""+H_STRATUM_WALLET+"."+H_STRATUM_PAYMENT_ID+"\",\"pass\":\""+H_STRATUM_PASSWORD+"\"}}\n";
			} else {
				  // set custom diff
					COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\""+H_STRATUM_WALLET+ "."+H_STRATUM_PAYMENT_ID+"+"+std::to_string(H_STRATUM_DIFF) + "\",\"pass\":\""+H_STRATUM_PASSWORD+"\"}}\n";
					std::cout << "DEBUG: " << COMMAND_LOGIN << std::endl;
			}

			//std::string COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\""+H_STRATUM_WALLET+"."+H_STRATUM_PAYMENT_ID+"\",\"pass\":\""+H_STRATUM_PASSWORD+"\"}}\n";
			//std::cout << COMMAND_LOGIN << std::endl;
			if (!stratumclient.send(COMMAND_LOGIN))
			  {
			    std::cout << log_time() << " [STRATUM] failed to send message: " << COMMAND_LOGIN << std::endl;
			    return false;
			  }
			  
			while (stratumclient.receive_ready(1000000))
			{
				std::string returned;
				if (!stratumclient.receive(returned)) {
					  std::cout << log_time() << " [STRATUM] receive failed" << std::endl;
					  return false;
				}
				else {
					if (returned.size()>20 && returned[0]=='{' && returned[returned.size()-2]=='}') {
						try {
							  	jsonxx::Object retval_json;
									retval_json.parse(returned);
									
									if (retval_json.has<jsonxx::Object>("result")) {
										jsonxx::Object retval_result = retval_json.get<jsonxx::Object>("result");
										jsonxx::Object job = retval_result.get<jsonxx::Object>("job");
										H_STRATUM_JOB_ID = job.get<jsonxx::String>("job_id");
										H_STRATUM_ID = job.get<jsonxx::String>("id");
										H_STRATUM_BLOB = job.get<jsonxx::String>("blob");
										H_STRATUM_TARGET = job.get<jsonxx::String>("target");
										stratum_connected = true;
									  	std::cout << log_time() << " [STRATUM] CONNECTED WITH ID " << H_STRATUM_ID << std::endl;
									} else {
										std::cout << log_time() << " [STRATUM] COULD NOT AUTHORIZE: " << returned << std::endl;
									}
						} catch (...) {
							  	//e
						}
					}
				}
			}
			
			return true;
		}
		
		// Stratum: getjob ---------------------------------------------------------------------------------------------
		bool stratum_getjob() {
			// stratumclient active?
			if (!stratumclient.initialised())
			      {
				std::cout << log_time() << " [STRATUM] connection has closed, reconnecting..." << std::endl;
				stratum_login();
			      }
			
			// get job info:
			std::string COMMAND_GETJOB = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getjob\",\"params\":{\"id\":\""+H_STRATUM_ID+"\"}}\n";
			if (!stratumclient.send(COMMAND_GETJOB))
			  {
			    std::cout << log_time() << " [STRATUM] failed to send message: " << COMMAND_GETJOB << std::endl;
			    ///// RECONNECT TO POOL:
			    stratum_login();
			    ///// ---
			    return false;
			  }
			  
		        while (stratumclient.receive_ready(2000000))
			{
				std::string returned = "";
				if (!stratumclient.receive(returned)) {
				  std::cout << log_time() << " [STRATUM] receive failed" << std::endl;
				  return false;
				}
				else {
						//std::cout << "DEBUG: getjob: " << returned << std::endl;
					  if (returned.size()>20 && returned[0]=='{' && returned[returned.size()-2]=='}') {
					  	  try {
								  	jsonxx::Object retval_json;
									  retval_json.parse(returned);
									
									  if (retval_json.has<jsonxx::Object>("result")) {
												jsonxx::Object retval_result = retval_json.get<jsonxx::Object>("result");
												H_STRATUM_JOB_ID = retval_result.get<jsonxx::String>("job_id");
												H_STRATUM_ID = retval_result.get<jsonxx::String>("id");
												H_STRATUM_BLOB = retval_result.get<jsonxx::String>("blob");
												H_STRATUM_TARGET = retval_result.get<jsonxx::String>("target");
											  //std::cout << log_time() << " [STRATUM] RETRIEVED JOB UPDATE JOB_ID = " << H_STRATUM_JOB_ID << std::endl;
									  } else {
											  //std::cout << log_time() << " [STRATUM] COULD NOT RETRIEVE JOB UPDATE: " << returned << std::endl;
									  }
							  } catch (...) {
							  	//e
							  }
					  }
				}
			}
			return true;
		}
		
		// Stratum: submit solution ------------------------------------------------------------------------------------------------
		bool stratum_submit_solution(std::string found_nonce, std::string found_hash) {
			// stratumclient active?
			if (!stratumclient.initialised())
			      {
				std::cout << log_time() << " [STRATUM] connection has closed, reconnecting..." << std::endl;
				stratum_login();
			      }
			
			// get job info:
			std::string COMMAND_SUBMIT = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"submit\",\"params\":{\"id\":\""+H_STRATUM_ID+"\",\"job_id\":\""+H_STRATUM_JOB_ID+"\",\"nonce\":\""+found_nonce+"\",\"result\":\""+found_hash+"\",\"algo\":\"dynexsolve\",\"mallob\":\""+H_MALLOB_NETWORK_ID+"\"}}\n";
			//std::cout << COMMAND_SUBMIT << std::endl;
			if (!stratumclient.send(COMMAND_SUBMIT))
			  {
			    std::cout << log_time() << " [STRATUM] failed to send message: " << COMMAND_SUBMIT << std::endl;
			    
			    ///// RECONNECT TO POOL:
			    stratum_login();
			    ///// ---

			    return false;
			  }
			  
		        while (stratumclient.receive_ready(2000000))
			{
				std::string returned;
				if (!stratumclient.receive(returned)) {
				  std::cout << log_time() << " [STRATUM] receive failed" << std::endl;
				  return false;
				}
				else {
					if (returned.size()>20 && returned[0]=='{' && returned[returned.size()-2]=='}') {
						try {
								//std::cout << "DEBUG: submitsolution: " << returned << std::endl;
								jsonxx::Object retval_json;
		            retval_json.parse(returned);
		            if (retval_json.has<jsonxx::Object>("result")) {
											int l = returned.size() - 2;
											returned.erase(l);
									  		//std::cout << log_time() << " [STRATUM] SHARE SUBMITTED. RESPONSE FROM POOL: \"" << returned << "\"" << std::endl;
											accepted_cnt++;
											std::cout << log_time() << TEXT_GREEN << " [STRATUM] SHARE ACCEPTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ")" << TEXT_DEFAULT << std::endl;

								} else {
											// really rejected? or just a job-update message
											if (!retval_json.has<jsonxx::Object>("params")) {
												rejected_cnt++;
												int l = returned.size() - 2;
												returned.erase(l);
												std::cout << log_time() << TEXT_RED << " [STRATUM] SHARE REJECTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ") WITH REASON " << TEXT_DEFAULT << returned << std::endl;
											}
								}
						} catch (...) {
							  	//e
						}
				  }

				}
			}
			// reconnect rest: stratum_login();
			return true;
		}
	
		// thread runner -----------------------------------------------------------------------------------------------------------
		bool hasher_work(int threadnum, int& sleep_time_ms, uint64_t& leffom, std::atomic<bool>& hasher_quit_flag, std::string daemon_host, std::string daemon_port, CURL* curl, std::string address, int reserve_size) {
			
			// init random generator:
			std::random_device rd;
			std::mt19937_64 gen(rd());
		    	std::uniform_int_distribution<uint64_t> dis;
			
			uint64_t height;
			uint64_t difficulty;
			uint64_t reward;
			float reward_dnx;
			
			std::string blockhashing_blob;
			block_template bt;
			block_header bh;
			
			if (!H_stratum) {
				// get block template:
				bt = get_blocktemplate(address, reserve_size, daemon_host, daemon_port, curl);
				blockhashing_blob = bt.blockhashing_blob; 
				height = bt.height;
				difficulty = bt.difficulty; 
				
				// last blockheader:
				bh = getlastblockheader(daemon_host, daemon_port, curl);
				reward = bh.reward;
				reward_dnx = (float)(reward)/1000000000;
			}

			// loop until quit:
			bool updatetemplate = true;
			while (!hasher_quit_flag) {
				// is there work?
				while (leffom > 0) {
					// check for new blocktemplate?
					if (updatetemplate || (hashcounter > 0 && hashcounter % 5000 == 0)) { // check every x hashes if still valid
						if (!H_stratum) {
							// last blockheader:
							bh = getlastblockheader(daemon_host, daemon_port, curl);
							bt = get_blocktemplate(address, reserve_size, daemon_host, daemon_port, curl);
							blockhashing_blob = bt.blockhashing_blob; 
							height = bt.height;
							difficulty = bt.difficulty; 
							reward = bh.reward;
							reward_dnx = (float)(reward)/1000000000;
						} else {
							//stratum:
							if (stratum_getjob()) {
								blockhashing_blob = H_STRATUM_BLOB;
								std::string diff_hex = H_STRATUM_TARGET.substr(6,2) + H_STRATUM_TARGET.substr(4,2) + H_STRATUM_TARGET.substr(2,2) + H_STRATUM_TARGET.substr(0,2);
									// convert target to difficulty:
								std::stringstream ss;
								ss << std::hex << diff_hex;
								ss >> difficulty;
								difficulty = (uint64_t)(4294967295 / difficulty);
								//std::cout << log_time() << " [STRATUM] new target " << H_STRATUM_TARGET << " hex " << diff_hex << " difficulty " << difficulty << std::endl;
							}
						}
						updatetemplate = false;	
					}
					// --- end check for new blocktemplate
					
					// check a chash
					uint64_t nonce = dis(gen);
					bool found = try_hash(blockhashing_blob, nonce, difficulty, false);
					
					// block found?
					if (found) {
						//std::cout << "Nonce found: " << nonce << std::endl;
						// convert nonce to hex:
						std::stringstream ss;
					    	ss << std::hex << std::setfill('0') << std::setw(sizeof(uint64_t)) << static_cast<int>(nonce);
					  
						std::string nonce_hex = ss.str().substr(6,2) + ss.str().substr(4,2) + ss.str().substr(2,2) + ss.str().substr(0,2); //little endian
						
						if (!H_stratum) {
							// add nonce into blocktemplate_blob:
              std::string submitdata = bt.blocktemplate_blob;
              submitdata.replace(78, 8, nonce_hex); 
							// submit block:
							bool validated = submitblock(submitdata, daemon_host, daemon_port, curl);
							if (validated) {
								accepted_cnt++;
								std::cout << log_time() << TEXT_GREEN << " [BLOCKCHAIN] *** BLOCK FOUND AND VALIDATED *** (difficulty " << difficulty << ", height " << height << ") (" << accepted_cnt << "/" << rejected_cnt << ")" << TEXT_DEFAULT << std::endl;
								
							} else {
								rejected_cnt++;
								std::cout << log_time() << " [BLOCKCHAIN] Block found but not accepted from node (" << accepted_cnt << "/" << rejected_cnt << ")" << std::endl;
								
							}
							// get block template:
							bt = get_blocktemplate(address, reserve_size, daemon_host, daemon_port, curl);
							blockhashing_blob = bt.blockhashing_blob; 
							height = bt.height;
							difficulty = bt.difficulty; 
							
							// last blockheader:
							bh = getlastblockheader(daemon_host, daemon_port, curl);
							reward = bh.reward;
							reward_dnx = (float)(reward)/1000000000;
						} else {
							// stratum submit:
							std::cout << log_time() << " [STRATUM] SUBMITTING PoW: " << H_POW << " (difficulty = " << difficulty << ")" << std::endl;
							if (stratum_submit_solution(nonce_hex, H_POW)){
								if (stratum_getjob()) {
									blockhashing_blob = H_STRATUM_BLOB;
									std::string diff_hex = H_STRATUM_TARGET.substr(6,2) + H_STRATUM_TARGET.substr(4,2) + H_STRATUM_TARGET.substr(2,2) + H_STRATUM_TARGET.substr(0,2);
									// convert target to difficulty:
									std::stringstream ss;
									ss << std::hex << diff_hex;
									ss >> difficulty;
									difficulty = (uint64_t)(4294967295 / difficulty);
									//std::cout << log_time() << " [STRATUM] new target " << H_STRATUM_TARGET << " hex " << diff_hex << " difficulty " << difficulty << std::endl;
								}
							}
						}
				
					}
					
					hashcounter++;
					//std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
					leffom--;
					if (leffom==0) {
						if (!H_stratum) {
							std::cout << log_time() << " [BLOCKCHAIN] UPDATED" << std::endl;
						} else {
							std::cout << log_time() << " [STRATUM] UPDATED" << std::endl;
						}
						updatetemplate = true;
					}
				}
				
			
			}
			// --- end loop
			
			std::cout << log_time() << " [BLOCKCHAIN] LOOPING STOPPED" << std::endl;
			hasher_quit_flag = true;
			
			return true;
		}
		// ----------------------------------------------------------------------------------------------------------------------------

};

namespace Dynexservice {
	class dynexservice {
		public:
			int num_threads;                         
			bool dynex_hasher_running = false;       // controls if another instance is already running (not allowed)
			std::atomic_bool dynex_hasher_quit_flag; // if true, thread stops (can be set from anywhere)
			uint64_t leffom = 0;	         
			int sleep_time_ms = 0; // unused
			CURL* curl;                              // CURL pointer
			
			
			bool init() {
				return true;
			}
			
			bool start(int threads_count, std::string daemon_host, std::string daemon_port, std::string address, int reserve_size, bool _stratum, std::string _STRATUM_URL, std::string _STRATUM_PORT, std::string _STRATUM_PAYMENT_ID, std::string _STRATUM_PASSWORD, int _STRATUM_DIFF, std::string _MALLOB_NETWORK_ID) {
				if (dynex_hasher_running) {
					std::cout << log_time() << " [BLOCKCHAIN] CANNOT START DYNEXSOLVE  SERVICE - ALREADY RUNNING" << std::endl;
			    		return false;	
				}	
				dynex_hasher_running = true;
				dynex_hasher_quit_flag = false;
				std::cout << log_time() << " [BLOCKCHAIN] STARTING DYNEXSOLVE SERVICE" << std::endl;
				/// STRATUM? /////////////////////////////////////////////////////////////////////////////////////////////////////
				if (_stratum) {
					// everything in global vars:
					H_STRATUM_WALLET = address;
					H_stratum = _stratum;
					H_STRATUM_URL = _STRATUM_URL;
					H_STRATUM_PORT = _STRATUM_PORT;
					H_STRATUM_PAYMENT_ID = _STRATUM_PAYMENT_ID;
					H_STRATUM_PASSWORD = _STRATUM_PASSWORD;
					H_STRATUM_DIFF = _STRATUM_DIFF;
					H_MALLOB_NETWORK_ID = _MALLOB_NETWORK_ID;
					std::cout << log_time() << " [STRATUM] CONNECTING TO " << H_STRATUM_URL << ":" << H_STRATUM_PORT << std::endl;
					stratumclient.initialise(H_STRATUM_URL.c_str(),(unsigned short)atoi(H_STRATUM_PORT.c_str()),5000000);
					if (!stratumclient.initialised())
					    {
					      std::cout << log_time() << " [STRATUM] client failed to initialise" << std::endl;
					      return -1;
					    }
					    if (stratumclient.error())
					    {
					      std::cout << log_time() << " [STRATUM] client initialisation failed with error " << stratumclient.error() << std::endl;
					      return -1;
					    }
					
					std::cout << log_time() << " [STRATUM] CONNECTED, LOGGING IN... " << std::endl;
					std::string COMMAND_LOGIN;
					if (H_STRATUM_DIFF==0) {
							COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\""+address+"."+H_STRATUM_PAYMENT_ID+"\",\"pass\":\""+H_STRATUM_PASSWORD+"\"}}\n";
					} else {
						  // set custom diff
							COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\""+address+ "."+H_STRATUM_PAYMENT_ID+"+"+std::to_string(H_STRATUM_DIFF) + "\",\"pass\":\""+H_STRATUM_PASSWORD+"\"}}\n";
							std::cout << "DEBUG: " << COMMAND_LOGIN << std::endl;
					}
					if (!stratumclient.send(COMMAND_LOGIN))
					  {
					    std::cout << log_time() << " [STRATUM] failed to send message: " << COMMAND_LOGIN << std::endl;
					    return -1;
					  }
					  
				        while (stratumclient.receive_ready(1000000))
					{
						std::string returned;
						if (!stratumclient.receive(returned)) {
						  std::cout << log_time() << " [STRATUM] receive failed" << std::endl;
						  return -1;
						}
						else {
						  	jsonxx::Object retval_json;
							retval_json.parse(returned);
							
							if (retval_json.has<jsonxx::Object>("result")) {
								jsonxx::Object retval_result = retval_json.get<jsonxx::Object>("result");
								jsonxx::Object job = retval_result.get<jsonxx::Object>("job");
								H_STRATUM_JOB_ID = job.get<jsonxx::String>("job_id");
								H_STRATUM_ID = job.get<jsonxx::String>("id");
								H_STRATUM_BLOB = job.get<jsonxx::String>("blob");
								H_STRATUM_TARGET = job.get<jsonxx::String>("target");
							  	stratum_connected = true;
							  	std::cout << log_time() << " [STRATUM] CONNECTED WITH ID " << H_STRATUM_ID << std::endl;
							} else {
								std::cout << log_time() << " [STRATUM] COULD NOT AUTHORIZE: " << returned << std::endl;
							}
						}
					}
				}
				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			    	// init curl: (for daemon)
			    	curl_global_init(CURL_GLOBAL_DEFAULT);
			    	
			    	// start chip threads:
			    	for (size_t i=0; i<threads_count; i++) {
			    		std::thread observer_th(dynex_hasher_thread_obj(), i, std::ref(sleep_time_ms), std::ref(leffom), std::ref(dynex_hasher_quit_flag), daemon_host, daemon_port, curl, address, reserve_size);
				    	observer_th.detach();
					assert(!observer_th.joinable());
			    	}
			    	
			    	curl_global_cleanup();
			    	
			    	return true;
			}
			
			bool stop() {
				dynex_hasher_running = false;
				dynex_hasher_quit_flag = true;
				std::cout << log_time() << " [BLOCKCHAIN] DYNEXSOLVE SERVICE STOPPED" << std::endl;
			    	return true;
			}
		private:
			// pretty log time:
			std::string log_time() {
			    auto t = std::time(nullptr);
			    auto tm = *std::localtime(&t);

			    std::ostringstream oss;
			    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
			    auto str = oss.str();

			    return str;
			}
	};
}










