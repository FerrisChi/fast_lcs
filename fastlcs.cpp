#include "fastlcs_cpu.h"

using namespace std;

P_ID_TYPE Record::cnt = 0;

P_ID_TYPE *id;

void FAST_LCS::init() {
  pairs = new Record[MAX_RECORD];
  level_pairs = new Record[MAX_RECORD];
  for (int i = 0; i < MAX_RECORD; i++) level_pairs[i] = Record();
  id = new P_ID_TYPE[MAX_RECORD];
}

void FAST_LCS::build_succT() {
  char *seq;
  int length;
  for (int i = 0; i < n_seq; i++) {
    seq = seqs[i];
    length = len_seq[i];

    vector<L_ID_TYPE> pos_label(NUM_LABEL + 1, INVALID_LID);
    for (L_ID_TYPE pos = length; pos >= 0; pos--) {
      for (int l = 1; l <= NUM_LABEL; l++) {
        succT[i][l][pos] = pos_label[l];
      }
      pos_label[CH[seq[pos]]] = pos;
    }
  }
}

void FAST_LCS::printT() {
  printf("Successor Table:\n");
  for (int i = 0; i < n_seq; i++) {
    printf("%s\n", seqs[i] + 1);
    for (int l = 1; l <= NUM_LABEL; l++) {
      for (int p = 0; p <= len_seq[i]; p++) printf("%ld ", succT[i][l][p]);
      printf("\n");
    }
    printf("\n");
  }
}

void printPair(Record &p, L_ID_TYPE len = Record::cnt) {
  printf("{ %ld ( ", p.id);
  for (int i = 0; i < len; i++) printf("%ld ", p.ipair[i]);
  printf(") %ld %ld %d } ", p.level, p.pred, p.state);
}

void printP(Record *pairs, P_ID_TYPE len) {
  printf("\n%ld pairs: ", len);
  for (int i = 0; i < len; i++) {
    printPair(pairs[i]);
  }
  printf("\n");
}

bool recordCmp(Record &a, Record &b) {
  int flag = 0;
  for (size_t i = 0; i < Record::cnt; i++) {
    if (a.ipair[i] < b.ipair[i])
      return 0;
    else if (a.ipair[i] > b.ipair[i])
      flag = 1;
  }
  return flag;
}

// bool recordCmp(Record &a, Record &b) {
//   for (size_t i = 0; i < Record::cnt; i++) {
//     if (a.ipair[i] < b.ipair[i]) return 0;
//   }
//   return 1;
// }

P_ID_TYPE prune(P_ID_TYPE r_size, Record *recordList) {
  // use j to prune i

  P_ID_TYPE level_cnt = 0, ans;
  for (P_ID_TYPE i = 0; i < r_size; i++)
    if (recordList[i].state != PAIR_STATE::INACTIVE) id[level_cnt++] = i;
  ans = level_cnt;

  for (P_ID_TYPE i = 0; i < level_cnt; i++) {
    if (recordList[id[i]].state == PAIR_STATE::INACTIVE) continue;
    for (P_ID_TYPE j = 0; j < level_cnt; j++) {
      if (i == j || recordList[id[j]].state == PAIR_STATE::INACTIVE) continue;
      if (recordCmp(recordList[id[i]], recordList[id[j]])) {
        // printPair(recordList[id[i]]);
        // printPair(recordList[id[j]]);
        // printf("\n");
        ans--;
        recordList[id[i]].state = PAIR_STATE::INACTIVE;
        break;
      }
    }
  }
  int p = 0;
  for (int i = 0; i < level_cnt; i++)
    if (recordList[id[i]].state == PAIR_STATE::ACTIVE) {
      recordList[p++] = recordList[id[i]];
    }
  return ans;
}

void FAST_LCS::get_init_ipair() {
  int flag;
  L_ID_TYPE *ipair = new L_ID_TYPE[n_seq];
  Record *p;
  for (int l = 1; l <= NUM_LABEL; l++) {
    flag = 1;
    for (int seq_i = 0; seq_i < n_seq; seq_i++) {
      if (succT[seq_i][l][0] == INVALID_LID) {
        flag = 0;
        break;
      }
      ipair[seq_i] = succT[seq_i][l][0];
    }
    if (flag) {
      pairs[pair_cnt] = Record(ipair, 1, INVALID_PID, pair_cnt);
      pair_cnt++;
    }
  }
  P_ID_TYPE cnt = prune(pair_cnt, pairs);
  for (P_ID_TYPE i = 0; i < cnt; i++) {
    pairs[i].id = i;
  }
  active_tl = active_size = pair_cnt = cnt;
}

void FAST_LCS::calc_LCS() {
  init();
  auto _outt =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] bg fastlcs " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
  // Step 1
  build_succT();
  // printT();

  // Step 2
  get_init_ipair();

  // Step 3
  P_ID_TYPE level_cnt;
  while (1) {
    // printf("%ld %ld %ld %ld\n", pair_cnt, active_hd, active_tl, active_size);
    // Step 3.1
    level_cnt = 0;
    L_ID_TYPE *succ = new L_ID_TYPE[MAX_N_SEQ];
    // printP(pairs, pair_cnt);
    for (P_ID_TYPE pid = active_hd; pid < pair_cnt; pid++) {
      Record &p = pairs[pid];
      // Step 3.1.1 Produce all the direct successors
      for (int l = 1; l <= NUM_LABEL; l++) {
        int flag = 1;
        for (int seq_i = 0; seq_i < n_seq; seq_i++) {
          if (succT[seq_i][l][p.ipair[seq_i]] == INVALID_LID) {
            flag = 0;
            break;
          }
          succ[seq_i] = succT[seq_i][l][p.ipair[seq_i]];
        }
        // Step 3.1.2
        if (flag) {
          level_pairs[level_cnt] = Record(succ, p.level + 1, p.id, INVALID_PID, PAIR_STATE::ACTIVE);
          level_cnt++;
        }
      }
      // Step 3.1.3
      p.state = PAIR_STATE::INACTIVE;
    }
    if (!level_cnt) {
      //   printf("Records in pairs are all inactive. break loop.\n");
      break;
    }
    // Step 3.2 pruning operations
    level_cnt = prune(level_cnt, level_pairs);
    active_hd = active_tl;
    for (int i = 0; i < level_cnt; i++) {
      level_pairs[i].id = active_tl;
      level_pairs[i].state = PAIR_STATE::INACTIVE;
      pairs[active_tl++] = level_pairs[i];
      pair_cnt++;
    }
    active_size = active_tl - active_hd;
  }

  // Step 4. Compute r = the maximum level
  lcs_len = pairs[pair_cnt - 1].level;
  lcs_cnt = active_size;
  for (P_ID_TYPE i = 0; i < active_size; i++) {
    // Step 4.1
    char *lcs = new char[lcs_len + 2];
    P_ID_TYPE pid = i + active_hd;
    while (pid != INVALID_PID) {
      Record &p = pairs[pid];
      lcs[p.level] = seqs[0][p.ipair[0]];
      pid = p.pred;
    }
    lcs[lcs_len + 1] = '\0';
    lcss.push_back(lcs);
  }
  _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] ed fastlcs " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001) << "\n";
}

void FAST_LCS::printLCS() {
  printf("Longest Common String length: %ld\n", lcs_len);
  printf("Possible %ld strings:\n", lcs_cnt);
  for (int i = 0; i < lcs_cnt; i++) {
    printf("%d: %s\n", i, lcss[i] + 1);
  }
}

int main(int argc, char *argv[]) {
  char input_path[100];
  strcpy(input_path, "/home/chijj/gpu/example/double/1.txt");
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--input") || !strcmp(argv[i], "-i")) {
      strcpy(input_path, argv[++i]);
    }
  }
  freopen(input_path, "r", stdin);
  int n;
  char seq[MAX_LENGTH + 2], *s;
  scanf("%d", &n);
  FAST_LCS lcs(n);
  getchar();
  seq[0] = 'x';
  for (int i = 0; i < lcs.n_seq; i++) {
    fgets(seq + 1, MAX_LENGTH, stdin);
    if (seq[strlen(seq) - 1] == '\n') {
      seq[strlen(seq) - 1] = '\0';
    }
    s = (char *)malloc((strlen(seq) + 1) * sizeof(char));
    strcpy(s, seq);
    lcs.seqs[i] = s;
    lcs.len_seq[i] = strlen(s + 1);
  }

  //   for (int i = 0; i < lcs.seqs.size(); i++) {
  //     printf("%ld %s\n", lcs.len_seq[i], lcs.seqs[i] + 1);
  //   }

  lcs.calc_LCS();
  printf("%ld\n%ld\n", lcs.lcs_len, lcs.lcs_cnt);
  // lcs.printLCS();

  return 0;
}
