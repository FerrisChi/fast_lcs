#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define BLOCK_THREAD_DIM 512

using namespace std;

typedef int64_t P_ID_TYPE;
typedef int64_t L_ID_TYPE;
typedef int64_t LEVEL_TYPE;

const L_ID_TYPE INVALID_LID = 0;
const P_ID_TYPE INVALID_PID = -1;
const L_ID_TYPE MAX_LENGTH = 1005;
const P_ID_TYPE MAX_RECORD = 5e6;
const int32_t MAX_N_SEQ = 15;
const int32_t NUM_LABEL = 4;

__constant__ L_ID_TYPE D_INVALID_LID = 0;
__constant__ P_ID_TYPE D_INVALID_PID = -1;
__constant__ int32_t D_NUM_LABEL = 4;
__constant__ P_ID_TYPE D_MAX_RECORD = 5e6;

// starts at value 1
unordered_map<char, int> CH = {{'A', 1}, {'C', 2}, {'G', 3}, {'T', 4}};

enum PAIR_STATE { ACTIVE, INACTIVE };

class Record {
 public:
  static P_ID_TYPE cnt;
  // start at index 0
  P_ID_TYPE id;
  L_ID_TYPE ipair[MAX_N_SEQ];
  LEVEL_TYPE level;
  P_ID_TYPE pred;
  PAIR_STATE state;

  Record() {
    id = level = pred = 0;
    state = PAIR_STATE::INACTIVE;
  }

  Record(L_ID_TYPE *ipair_, LEVEL_TYPE level_, P_ID_TYPE pred_, P_ID_TYPE id_ = INVALID_PID,
         PAIR_STATE state_ = PAIR_STATE::ACTIVE)
      : id(id_), level(level_), pred(pred_), state(state_) {
    // memcpy(ipair, ipair_, cnt * sizeof(L_ID_TYPE));
    for (int i = 0; i < cnt; i++) ipair[i] = ipair_[i];
    // state = PAIR_STATE::ACTIVE;
  }
  Record &operator=(const Record &r) {
    this->id = r.id;
    for (int i = 0; i < cnt; i++) this->ipair[i] = r.ipair[i];
    // memcpy(this->ipair, r.ipair, MAX_N_SEQ * sizeof(L_ID_TYPE));
    this->level = r.level;
    this->pred = r.pred;
    this->state = r.state;
    return *this;
  }
};

class FAST_LCS {
 public:
  int n_seq;
  // start at index 1
  char *h_seqs;
  char *d_seqs;
  L_ID_TYPE lens[MAX_N_SEQ];
  L_ID_TYPE max_len;
  Record *a_pairs;
  Record *m_pairs;
  Record *a_level_pairs;
  Record *m_level_pairs;

  L_ID_TYPE *h_succT;
  L_ID_TYPE *d_succT;
  L_ID_TYPE Tx, Ty, Tz;

  P_ID_TYPE pair_cnt;
  P_ID_TYPE active_size;
  P_ID_TYPE active_hd;
  P_ID_TYPE active_tl;

  L_ID_TYPE lcs_len;
  L_ID_TYPE lcs_cnt;
  char *h_lcss;
  char *d_lcss;

  FAST_LCS(){};
  FAST_LCS(int n_seq_) : n_seq(n_seq_) {
    Record::cnt = n_seq;
    max_len = active_hd = active_tl = active_size = pair_cnt = lcs_len = lcs_cnt = 0;
  };
  void init();
  void build_succT();
  void get_init_ipair();
  void calc_LCS();

  void printT();
  void printLCS();
};