#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    level = 0;
    id = pred = INVALID_PID;
    state = PAIR_STATE::INACTIVE;
  }

  Record(L_ID_TYPE *ipair_, LEVEL_TYPE level_, P_ID_TYPE pred_, P_ID_TYPE id_ = INVALID_PID,
         PAIR_STATE state_ = PAIR_STATE::ACTIVE)
      : id(id_), level(level_), pred(pred_), state(state_) {
    memcpy(ipair, ipair_, MAX_N_SEQ * sizeof(L_ID_TYPE));
  }
};

class FAST_LCS {
 public:
  int n_seq;
  // start at index 1
  char *seqs[MAX_N_SEQ];
  L_ID_TYPE len_seq[MAX_N_SEQ];
  Record *pairs;
  Record *level_pairs;
  P_ID_TYPE pair_cnt;
  P_ID_TYPE active_hd;
  P_ID_TYPE active_tl;
  P_ID_TYPE active_size;

  L_ID_TYPE succT[MAX_N_SEQ][NUM_LABEL + 1][MAX_LENGTH];
  vector<char *> lcss;
  L_ID_TYPE lcs_len;
  L_ID_TYPE lcs_cnt;

  FAST_LCS(){};
  FAST_LCS(int n_seq_) : n_seq(n_seq_) {
    Record::cnt = n_seq;
    active_hd = active_tl = active_size = pair_cnt = lcs_len = lcs_cnt = 0;
  }

  void init();
  void build_succT();
  void printT();
  void get_init_ipair();
  void calc_LCS();
  void printLCS();
};
