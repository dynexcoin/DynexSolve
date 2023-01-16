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
	#define __builtin_bswap32(x) _byteswap_ulong(x)
	#define NOMINMAX
#endif // WIN32

extern "C" void cn_slow_hash(const void* data, size_t length, uint8_t * hash);
extern "C" void slow_hash_allocate_state(void);
extern "C" void slow_hash_free_state(void);

extern std::string VERSION;

//#define CURL_STATICLIB
#include <curl/curl.h> //required for MPI - dependency
#include "jsonxx.h"    //no install required + cross platform - https://github.com/hjiang/jsonxx
// stratum:
#include "tcp_sockets.hpp" // - https://stlplus.sourceforge.net/stlplus3/docs/tcp_sockets.html
#include "log.hpp"

static std::string H_STRATUM_JOB_ID = "";
static std::string H_STRATUM_ID = "";
static std::string H_STRATUM_TARGET = "";
static bool        H_stratum = false;
static std::string H_STRATUM_URL = "";
static int         H_STRATUM_PORT = 0;
static std::string H_STRATUM_USER = "";
static std::string H_STRATUM_PASSWORD = "";
static std::string H_MALLOB_NETWORK_ID = "";
static uint64_t    H_DIFF = 0;
static uint64_t    H_POW_DIFF = 0;
static std::string H_POW = "";
static uint32_t    H_NONCE = 0;
static uint8_t     H_BLOB[256] = {0}; // 76
static size_t      H_BLOB_SIZE = 0;
static stlplus::TCP_client stratumclient;
static uint64_t hashcounter = 0;
static uint32_t accepted_cnt = 0;
static uint32_t rejected_cnt = 0;

static std::string POUW_BLOB = "";
static std::string POUW_HASH = "";
static std::string POUW_DIFF = "";
static std::string POUW_JOB = "";

static int H_STRATUM_INTERVAL = 100;

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
// convert string to uint8_t:
static inline void convert(const char *s, uint8_t * out, int maxsize) {
	int i = 0;
	while (*s && i < maxsize) {
		char byte[3] = { *s, *(s + 1), 0 };
		out[i++] = strtol(byte, NULL, 16);
		s += 2;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynex Service Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class dynex_hasher_thread_obj {

	public:

		// operator function --------------------------------------------------------------------------------------------
		void operator()(int thread_id, uint64_t& leffom, std::atomic<bool>& hasher_quit_flag, std::string daemon_host, std::string daemon_port, CURL* curl, std::string address, int reserve_size)
		{
			bool workfinished = hasher_work(thread_id, leffom, hasher_quit_flag, daemon_host, daemon_port, curl, address, reserve_size);
			return;
		}
		// ---------------------------------------------------------------------------------------------------------------

	private:
		int id = 0;
		std::vector<int> pending;
		std::string buffer = "";
		uint64_t recvcounter = 0;

		// curl return value function:
		static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp){
			((std::string*)userp)->append((char*)contents, size * nmemb);
			return size * nmemb;
		}

		// try_hash: ----------------------------------------------------------------------------------------
		bool try_hash(uint32_t nonce, bool logging = false) {
			if (H_DIFF == 0) {
				return false;
			}

			if (logging) {
				std::stringstream ss;
				ss << std::hex << std::setfill('0') << std::setw(sizeof(uint64_t)) << static_cast<int>(__builtin_bswap32(nonce));
				std::cout << "nonce_hex: " << ss.str() << std::endl;
				std::cout << "input_hex: ";
				for (int i=0; i < H_BLOB_SIZE; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>((uint8_t)H_BLOB[i]);
				std::cout << std::endl;
			}

			memcpy(&H_BLOB[39], &nonce, 4); // add nonce into blob => input

			uint8_t hash[32];
			cn_slow_hash(H_BLOB, H_BLOB_SIZE, hash);

			H_POW_DIFF = *(uint32_t*)&hash[28] ? (uint64_t)(4294967295 / *(uint32_t*)&hash[28]) : 4294967296;

			// nonce found?
			if (H_POW_DIFF >= H_DIFF) {
				//std::cout << " debug: check_hash " << outhash << " with difficulty " << difficulty << " result: TRUE" << std::endl;
				// convert output hash to std::string:
				std::stringstream _outhash; _outhash << std::hex << std::setfill('0');
				for (int i = 0; i < 32; i++) _outhash << std::hex << std::setw(2) << static_cast<int>((uint8_t)hash[i]);
				std::string outhash = _outhash.str();
				if (logging) std::cout << "OUTPUT (std::string) : " << outhash << std::endl;
				if (H_stratum) {
					H_POW = outhash; // stratum needs proof of work
				}
				return true;
			}

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
			std::string postfields = "{\"jsonrpc\":\"2.0\",\"method\":\"submitblock\",\"params\":[\""+blockdata+"\"]}";
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

		uint32_t gen_nonce() {
			static std::random_device rd;
			static std::mt19937 gen(rd());
			static std::uniform_int_distribution<uint32_t> dis;
			return dis(gen);
		}

		// Stratum: login ----------------------------------------------------------------------------------------------
		bool stratum_login() {
			static bool wait = false;
			id = 0;
			buffer = "";
			pending.clear();
			H_DIFF = 0; // stop hashing
			//accepted_cnt = 0;
			//rejected_cnt = 0;

			// reconnect properly
			if (stratumclient.initialised()) {
				stratumclient.close();
			}

			// wait before reconnect
			if (wait) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5000 + rand()%5000));
			} else {
				wait = true;
			}

			LogTS << "[STRATUM] CONNECTING TO " << H_STRATUM_URL << ":" << H_STRATUM_PORT << std::endl;
			stlplus::TCP_client client(H_STRATUM_URL.c_str(), H_STRATUM_PORT, 10000000);
			stratumclient = client;

			if (!stratumclient.initialised()) {
				LogTS << "[STRATUM] client failed to initialise" << std::endl;
				return false;
			}

			if (stratumclient.error()) {
				LogTS << " [STRATUM] client initialisation failed with error " << stratumclient.error() << std::endl;
				return false;
			}

			LogTS << "[STRATUM] CONNECTED, LOGGING IN... " << std::endl;

			std::string COMMAND_LOGIN;
			COMMAND_LOGIN = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(++id) + ",\"method\":\"login\",\"params\":{\"login\":\""+H_STRATUM_USER+"\",\"pass\":\""+H_STRATUM_PASSWORD+"\",\"agent\":\""+VERSION+"\"}}\n";
			//std::cout << "### " << COMMAND_LOGIN << std::endl;

			if (!stratumclient.send(COMMAND_LOGIN)) {
				LogTS << "[STRATUM] failed to send message: " << COMMAND_LOGIN << std::endl;
				return false;
			}

			return stratum_recv(1);
		}

		// Stratum: parse ---------------------------------------------------------------------------------------------
		bool stratum_parse(std::string input) {
			//LogTS << "[STRATUM] PARSE: " << input << std::endl;

			jsonxx::Object retval_json;
			if (!retval_json.parse(input)) {
				int size = buffer.size();
				buffer.append(input);
				if (!size) return true;
				if (!retval_json.parse(buffer)) {
					LogTS << "[STRATUM] JSON PARSE ERROR: " << buffer << std::endl;
					buffer = input;
					return true;
				}
				buffer = "";
			}
			if (buffer != "") {
				LogTS << "[STRATUM] JSON PARSE ERROR: " << buffer << std::endl;
				buffer = "";
			}

			int retval_id = retval_json.has<jsonxx::Number>("id")?retval_json.get<jsonxx::Number>("id"):0;
			auto share = std::find(pending.begin(), pending.end(), retval_id);

			if (retval_json.has<jsonxx::Object>("error")) {
				if (share != pending.end()) {
					pending.erase(share);
					rejected_cnt++;
					LogTS << "[STRATUM] " << TEXT_RED << "SHARE REJECTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ") WITH REASON: " << stratum_geterror(retval_json) << TEXT_DEFAULT << std::endl;
					// last share
					if (id == retval_id) {
						// assume pending shares are lost so count them as rejected
						if (pending.size()) {
							rejected_cnt += pending.size();
							pending.clear();
							LogTS << "[STRATUM] " << TEXT_RED << "SHARE REJECTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ") WITH NO REPLY" << TEXT_DEFAULT << std::endl;
						}
						return stratum_getjob(); // force job update
					}
				} else {
					LogTS << TEXT_RED << "[STRATUM] ERROR: " << stratum_geterror(retval_json) << TEXT_DEFAULT << std::endl;
					if (id == retval_id && id > 1) {
						stratum_login();
						return false;
					}
				}
			} else if (retval_json.has<jsonxx::Object>("params")) {
				// getjob answer
				return stratum_newjob(retval_json.get<jsonxx::Object>("params"));
			} else if (retval_json.has<jsonxx::Object>("result")) {
				if (retval_json.get<jsonxx::Object>("result").has<jsonxx::Object>("job")) {
					LogTS << TEXT_SILVER << "[STRATUM] AUTHORIZED" << TEXT_DEFAULT << std::endl;
					stratum_newjob(retval_json.get<jsonxx::Object>("result").get<jsonxx::Object>("job"));
				} else if (share != pending.end()) {
					pending.erase(share);
					accepted_cnt++;
					LogTS << "[STRATUM] " << TEXT_GREEN << "SHARE ACCEPTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ")" << TEXT_DEFAULT << std::endl;
					// last share
					if (id == retval_id) {
						// assume pending shares are lost so count them as rejected
						if (pending.size()) {
							rejected_cnt += pending.size();
							pending.clear();
							LogTS << "[STRATUM] " << TEXT_RED << "SHARE REJECTED BY POOL (" << accepted_cnt << "/" << rejected_cnt << ") WITH NO REPLY" << TEXT_DEFAULT << std::endl;
						}
					}
				} else {
					stratum_newjob(retval_json.get<jsonxx::Object>("result"));
				}
			} else {
				LogTS << "[STRATUM] UNKNOWN: " << retval_json.json() << std::endl;
			}
			return true;
		}

		// Stratum: recv ---------------------------------------------------------------------------------------------
		bool stratum_recv(int wait) {
			recvcounter = hashcounter;
			// stratumclient active?
			if (!stratumclient.initialised()) {
				LogTS << "[STRATUM] connection has closed, reconnecting..." << std::endl;
				stratum_login();
				return false;
			}

			while (stratumclient.receive_ready(wait?2000000:1)) {
				std::string returned = "";
				if (!stratumclient.receive(returned)) {
					LogTS << "[STRATUM] receive failed" << std::endl;
					stratum_login();
					return false;
				}

				//LogTS << "[STRATUM] RECV: " << returned << std::endl;
				//std::cout << "---------------------------------------------------------------------------------" << std::endl;
				//std::cout << "H_STRATUM_INTERVAL = " << H_STRATUM_INTERVAL << std::endl;
				//std::cout << TEXT_GREEN << returned << TEXT_DEFAULT << std::endl;
				//std::cout << "---------------------------------------------------------------------------------" << std::endl;

				// #499 stale share fix (some pools return multiple jobs, only last one is valid)
				std::stringstream returned_ss(returned);
				for (std::string line; std::getline(returned_ss, line, '\n');) {
					//std::cout << TEXT_RED << "PARSED: " << line << TEXT_DEFAULT << std::endl;
					if (!stratum_parse(line))
						return false;
				}
			}
			return true;
		}

		// Stratum: getjob ---------------------------------------------------------------------------------------------
		bool stratum_getjob() {
			// stratumclient active?
			if (!stratumclient.initialised()) {
				LogTS << "[STRATUM] connection has closed, reconnecting..." << std::endl;
				stratum_login();
				return false;
			}

			//LogTS << "[STRATUM] GET JOB " << H_STRATUM_JOB_ID << std::endl;
			std::string COMMAND_GETJOB = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(++id) + ",\"method\":\"getjob\",\"params\":{\"id\":\""+H_STRATUM_ID+"\"}}\n";
			if (!stratumclient.send(COMMAND_GETJOB)) {
				LogTS << "[STRATUM] failed to send message: " << COMMAND_GETJOB << std::endl;
				stratum_login();
				return false;
			}

			return stratum_recv(1);
		}

		// Stratum: submit solution ------------------------------------------------------------------------------------------------
		bool stratum_submit_solution(std::string found_nonce, std::string found_hash) {
			// stratumclient active?
			if (!stratumclient.initialised()) {
				LogTS << "[STRATUM] connection has closed, reconnecting..." << std::endl;
				stratum_login();
				return false;
			}

			std::string COMMAND_SUBMIT = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(++id) + ",\"method\":\"submit\",\"params\":{\"id\":\""+H_STRATUM_ID+"\",\"job_id\":\""+H_STRATUM_JOB_ID+"\",\"nonce\":\""+found_nonce+"\",\"result\":\""+found_hash+"\",\"algo\":\"dynexsolve\",\"mallob\":\""+H_MALLOB_NETWORK_ID+"\",\"POUW_JOB\":\""+POUW_JOB+"\",\"POUW_BLOB\":\""+POUW_BLOB+"\",\"POUW_HASH\":\""+POUW_HASH+"\",\"POUW_DIFF\":\""+POUW_DIFF+"\"}}\n";
			LogTS << "[STRATUM] SUBMITTING PoW: " << H_POW << " NONCE " << found_nonce << " DIFF " << H_POW_DIFF << std::endl;

			//std::cout << COMMAND_SUBMIT << std::endl;
			if (!stratumclient.send(COMMAND_SUBMIT)) {
				LogTS << "[STRATUM] failed to send message: " << COMMAND_SUBMIT << std::endl;
				stratum_login();
				return false;
			}

			pending.push_back(id);
			return stratum_recv(0);
		}

		// Stratum: geterror ---------------------------------------------------------------------------------------------
		std::string stratum_geterror(jsonxx::Object obj) {
			if (obj.has<jsonxx::Object>("error") && obj.get<jsonxx::Object>("error").has<jsonxx::String>("message")) {
				return obj.get<jsonxx::Object>("error").get<jsonxx::String>("message");
			}
			return obj.json();
		}

		// Stratum: newjob ---------------------------------------------------------------------------------------------
		bool stratum_newjob(jsonxx::Object obj) {
			if (!obj.has<jsonxx::String>("job_id") ||
				!obj.has<jsonxx::String>("id") ||
				!obj.has<jsonxx::String>("blob") ||
				!obj.has<jsonxx::String>("target")) {
				LogTS << "[STRATUM] JOB PARSE ERROR: " << obj.json() << std::endl;
				return false;
			}

			//LogTS << "[STRATUM] " << "JOB: " << obj.json() << std::endl;
			H_STRATUM_JOB_ID = obj.get<jsonxx::String>("job_id");
			H_STRATUM_ID = obj.get<jsonxx::String>("id");
			H_STRATUM_TARGET = obj.get<jsonxx::String>("target");

			// convert target to difficulty:
			uint64_t targ;
			std::stringstream ss;
			ss << std::hex << H_STRATUM_TARGET;
			ss >> targ;
			H_DIFF = targ ? (uint64_t)(4294967295 / __builtin_bswap32(targ & 0xFFFFFFFF)) : 0;
			LogTS << "[STRATUM] " << "JOB ID " << H_STRATUM_JOB_ID << " DIFF " << H_DIFF << std::endl;

			std::string blob = obj.get<jsonxx::String>("blob");
			H_BLOB_SIZE = std::min(blob.size()/2, sizeof(H_BLOB));
			// convert hex blob to uint8_t
			convert(blob.c_str(), H_BLOB, H_BLOB_SIZE);

			return true;
		}

		// thread runner -----------------------------------------------------------------------------------------------------------
		bool hasher_work(int threadnum, uint64_t& leffom, std::atomic<bool>& hasher_quit_flag, std::string daemon_host, std::string daemon_port, CURL* curl, std::string address, int reserve_size) {

			block_template bt;
			//block_header bh;

			H_DIFF = 0;
			H_NONCE = gen_nonce();

			while (!leffom && !hasher_quit_flag) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

			if (H_stratum) {
				stratum_login();
			}

			slow_hash_allocate_state();

			// loop until quit:
			bool updatetemplate = true;
			while (!hasher_quit_flag) {

				// is there work?
				while (leffom > 0) {
					// check for new blocktemplate?
					if (updatetemplate || hashcounter - recvcounter > H_STRATUM_INTERVAL || H_DIFF == 0) { // check every x hashes if still valid
						if (!H_stratum) {
							// last blockheader:
							bt = get_blocktemplate(address, reserve_size, daemon_host, daemon_port, curl);
							H_DIFF = bt.difficulty;
							H_BLOB_SIZE = std::min(bt.blockhashing_blob.size()/2, sizeof(H_BLOB));
							// convert hex blob to uint8_t
							convert(bt.blockhashing_blob.c_str(), H_BLOB, H_BLOB_SIZE);
							//bh = getlastblockheader(daemon_host, daemon_port, curl);
							H_NONCE = gen_nonce();
						} else {
							//stratum:
							stratum_recv(0);
							if (!H_DIFF) {
								stratum_getjob();
							}
						}
						if (!H_DIFF) {
							LogTS << "[STRATUM] NO JOB, WAITING " << std::endl;
							std::this_thread::sleep_for(std::chrono::milliseconds(3000 + rand()%3000));
							continue;
						}
						updatetemplate = false;
					}

					// check a chash
					bool found = try_hash(H_NONCE, false);

					// block found?
					if (found) {
						//std::cout << "Nonce found: " << nonce << std::endl;
						// convert nonce to hex:
						std::stringstream ss;
						ss << std::hex << std::setfill('0') << std::setw(sizeof(uint64_t)) << static_cast<int>(__builtin_bswap32(H_NONCE));
						std::string nonce_hex = ss.str();
						if (!H_stratum) {
							// add nonce into blocktemplate_blob:
							std::string submitdata = bt.blocktemplate_blob;
							submitdata.replace(78, 8, nonce_hex);
							// submit block:
							bool validated = submitblock(submitdata, daemon_host, daemon_port, curl);
							if (validated) {
								accepted_cnt++;
								LogTS << TEXT_GREEN << "[BLOCKCHAIN] *** BLOCK FOUND AND VALIDATED *** (difficulty " << bt.difficulty << ", height " << bt.height << ") (" << accepted_cnt << "/" << rejected_cnt << ")" << TEXT_DEFAULT << std::endl;
							} else {
								rejected_cnt++;
								LogTS << "[BLOCKCHAIN] Block found but not accepted from node (" << accepted_cnt << "/" << rejected_cnt << ")" << std::endl;
							}
						} else {
							// stratum submit:
							stratum_submit_solution(nonce_hex, H_POW);
						}
						updatetemplate = true;
					}

					H_NONCE++;
					if (H_NONCE > 4294967295) H_NONCE = 0; // avoid overflow error
					hashcounter++;
					leffom--;
					if (leffom == 0) {
						if (!H_stratum) {
							LogTS << "[BLOCKCHAIN] UPDATED" << std::endl;
						} else {
							stratum_recv(0);
							LogTS << "[STRATUM] UPDATED" << std::endl;
						}
						updatetemplate = true;
					}
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(10));

			}
			// --- end loop

			LogTS << "[BLOCKCHAIN] LOOPING STOPPED" << std::endl;
			hasher_quit_flag = true;
			slow_hash_free_state();
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
			CURL* curl;                              // CURL pointer

			bool init() {
				return true;
			}

			bool start(int threads_count, std::string daemon_host, std::string daemon_port, std::string user, int reserve_size, bool stratum, std::string url, int port, std::string pass, std::string MALLOB_NETWORK_ID) {
				if (dynex_hasher_running) {
					LogTS << "[BLOCKCHAIN] CANNOT START DYNEXSOLVE  SERVICE - ALREADY RUNNING" << std::endl;
						return false;
				}
				dynex_hasher_running = true;
				dynex_hasher_quit_flag = false;
				LogTS << "[BLOCKCHAIN] STARTING DYNEXSOLVE SERVICE" << std::endl;
				/// STRATUM? /////////////////////////////////////////////////////////////////////////////////////////////////////
				if (stratum) {
					// everything in global vars:
					H_stratum = stratum;
					H_STRATUM_URL = url;
					H_STRATUM_PORT = port;
					H_STRATUM_PASSWORD = pass;
					H_STRATUM_USER = user;
					H_MALLOB_NETWORK_ID = MALLOB_NETWORK_ID;
				}
				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				// init curl: (for daemon)
				curl_global_init(CURL_GLOBAL_DEFAULT);

				// start chip threads:
				for (size_t i=0; i<threads_count; i++) {
					std::thread observer_th(dynex_hasher_thread_obj(), i, std::ref(leffom), std::ref(dynex_hasher_quit_flag), daemon_host, daemon_port, curl, user, reserve_size);
					observer_th.detach();
					assert(!observer_th.joinable());
				}

				curl_global_cleanup();
				return true;
			}

			bool stop() {
				dynex_hasher_running = false;
				dynex_hasher_quit_flag = true;
				LogTS << "[BLOCKCHAIN] DYNEXSOLVE SERVICE STOPPED" << std::endl;
				return true;
			}

			void getstats(uint64_t* hashes, uint32_t* accepted, uint32_t* rejected) {
				*hashes = hashcounter;
				*accepted = accepted_cnt;
				*rejected = rejected_cnt;
			}

	};
}
