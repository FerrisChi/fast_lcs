#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "fastlcs_cu.h"
using namespace std;

#define CHK_CU(x)                             \
  if (is_cuda_error((x), __LINE__) == true) { \
    assert(false);                            \
  }

bool is_cuda_error(cudaError_t error, int line) {
  // cudaError_t error = cudaGetLastError ();
  if (error != cudaSuccess) {
    const char *error_string = cudaGetErrorString(error);
    std::cout << "Cuda Error: " << error_string << " " << line << std::endl;
    return true;
  }
  return false;
}

P_ID_TYPE Record::cnt = 0;
bool *a_flag, *m_flag;

// find index of A[i][j][k] in A[x][y][z]
__device__ __host__ inline int get_id(int i, int j, L_ID_TYPE k, L_ID_TYPE x, L_ID_TYPE y, L_ID_TYPE z) {
  return i * y * z + j * z + k;
}

// find index of A[i][j][k] in A[x][y][z]
__device__ __host__ inline int get_str_hd(int i, L_ID_TYPE len) { return i * len; }

void FAST_LCS::printT() {
  printf("Successor Table:\n");
  for (int i = 0; i < n_seq; i++) {
    printf("%s\n", h_seqs + get_str_hd(i, max_len + 2) + 1);
    for (int l = 1; l <= NUM_LABEL; l++) {
      int len = lens[i];
      for (L_ID_TYPE p = 0; p <= len; p++) {
        printf("%ld ", h_succT[get_id(i, l, p, Tx, Ty, Tz)]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

__host__ __device__ void printPair(Record &p, L_ID_TYPE len) {
  printf("{ %ld ( ", p.id);
  for (int i = 0; i < len; i++) printf("%ld ", p.ipair[i]);
  printf(") %ld %ld %d } ", p.level, p.pred, p.state);
}

void printP(Record *pairs, P_ID_TYPE len) {
  printf("\n%ld pairs: ", len);
  for (int i = 0; i < len; i++) {
    printPair(pairs[i], Record::cnt);
  }
  printf("\n");
}

void FAST_LCS::printLCS() {
  printf("Longest Common String length: %ld\n", strlen(h_lcss + 1));
  printf("Possible %ld strings:\n", active_size);
  for (int i = 0; i < active_size; i++) {
    printf("%d: %s\n", i, h_lcss + get_str_hd(i, lcs_len + 2) + 1);
  }
}

void FAST_LCS::init() {
  Tx = n_seq;
  Ty = NUM_LABEL + 1;
  Tz = max_len + 1;

  cudaSetDeviceFlags(cudaDeviceMapHost);

  CHK_CU(cudaMalloc(&d_seqs, (max_len + 2) * n_seq * sizeof(char)));
  CHK_CU(cudaMemcpy(d_seqs, h_seqs, (max_len + 2) * n_seq * sizeof(char), cudaMemcpyHostToDevice));

  CHK_CU(cudaHostAlloc(&a_pairs, MAX_RECORD * sizeof(Record), cudaHostAllocMapped));
  CHK_CU(cudaHostGetDevicePointer(&m_pairs, a_pairs, 0));

  h_succT = new L_ID_TYPE[Tx * Ty * Tz];
  CHK_CU(cudaMalloc(&d_succT, Tx * Ty * Tz * sizeof(L_ID_TYPE)));

  a_flag = new bool[MAX_RECORD];
  CHK_CU(cudaHostAlloc(&a_flag, MAX_RECORD * sizeof(bool), cudaHostAllocMapped));
  CHK_CU(cudaHostGetDevicePointer(&m_flag, a_flag, 0));

  CHK_CU(cudaHostAlloc(&a_level_pairs, MAX_RECORD * sizeof(Record), cudaHostAllocMapped));
  for (int i = 0; i < MAX_RECORD; i++) a_level_pairs[i] = Record();
  CHK_CU(cudaHostGetDevicePointer(&m_level_pairs, a_level_pairs, 0));
  CHK_CU(cudaDeviceSynchronize());
  // printf("Initialized. Tx: %ld, Ty: %ld, Tz: %ld\n", Tx, Ty, Tz);
}

void FAST_LCS::build_succT() {
  int length;
  L_ID_TYPE *pos_label = new L_ID_TYPE[NUM_LABEL + 1];
  for (int i = 0; i < n_seq; i++) {
    length = lens[i];
    for (int l = 1; l <= NUM_LABEL; l++) {
      pos_label[l] = INVALID_LID;
    }
    for (L_ID_TYPE pos = length; pos >= 0; pos--) {
      for (int l = 1; l <= NUM_LABEL; l++) {
        h_succT[get_id(i, l, pos, Tx, Ty, Tz)] = pos_label[l];
      }
      if (pos) pos_label[CH[h_seqs[get_str_hd(i, max_len + 2) + pos]]] = pos;
    }
  }
  CHK_CU(cudaMemcpyAsync(d_succT, h_succT, Tx * Ty * Tz * sizeof(L_ID_TYPE), cudaMemcpyHostToDevice));
  delete pos_label;
}

// whether can use b to prune a
__device__ bool recordCmp(Record &a, Record &b, int cnt) {
  int flag = 0;
  for (size_t i = 0; i < cnt; i++) {
    if (a.ipair[i] < b.ipair[i])
      return 0;
    else if (a.ipair[i] > b.ipair[i])
      flag = 1;
  }
  return flag;
}

// use only to calculate LCS length
// bool recordCmp(Record &a, Record &b) {
//   for (size_t i = 0; i < Record::cnt; i++) {
//     if (a.ipair[i] < b.ipair[i]) return 0;
//   }
//   return 1;
// }

__global__ void prune(int n_seq, P_ID_TYPE max_size, bool *flag, Record *level_pairs) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= max_size) return;
  flag[thread_id] = (level_pairs[thread_id].state == PAIR_STATE::ACTIVE);
  if (!flag[thread_id]) return;
  for (P_ID_TYPE i = 0; i < max_size; i++) {
    if (i == thread_id || level_pairs[i].state == PAIR_STATE::INACTIVE) continue;
    if (recordCmp(level_pairs[thread_id], level_pairs[i], n_seq)) {
      flag[thread_id] = false;
      break;
    }
  }
}

void FAST_LCS::get_init_ipair() {
  int flag;
  L_ID_TYPE *ipair = new L_ID_TYPE[n_seq];
  for (int l = 1; l <= NUM_LABEL; l++) {
    flag = 1;
    for (int i = 0; i < n_seq; i++) {
      if (h_succT[get_id(i, l, 0, Tx, Ty, Tz)] == INVALID_LID) {
        flag = 0;
        break;
      }
      ipair[i] = h_succT[get_id(i, l, 0, Tx, Ty, Tz)];
    }
    if (!flag) continue;
    a_pairs[pair_cnt] = Record(ipair, 1, INVALID_PID, pair_cnt, PAIR_STATE::ACTIVE);
    pair_cnt++;
  }
  delete ipair;

  prune<<<1, pair_cnt>>>(n_seq, pair_cnt, m_flag, m_pairs);
  CHK_CU(cudaDeviceSynchronize());

  for (P_ID_TYPE i = 0; i < pair_cnt; i++) {
    if (!a_flag[i]) continue;
    a_pairs[i].id = active_tl;
    a_pairs[i].state = PAIR_STATE::INACTIVE;
    a_pairs[active_tl++] = a_pairs[i];
  }
  active_size = pair_cnt = active_tl;
}

template <class T>
inline T thread_block_size(const T total, const T tb_size) {
  if (total % tb_size == 0) return total / tb_size;
  return total / tb_size + 1;
}

__global__ void find_dir_pair(int n_seq, L_ID_TYPE *succT, Record *new_pairs, Record *pairs, P_ID_TYPE ac_size,
                              P_ID_TYPE ac_hd, P_ID_TYPE ac_tl, L_ID_TYPE Tx, L_ID_TYPE Ty, L_ID_TYPE Tz) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= ac_size) return;
  int valid_cnt = 0;
  int flag;
  L_ID_TYPE p;

  Record nowpair = pairs[ac_hd + thread_id];
  // if(!thread_id) {
  //     printf("now id: %ld\n", ac_hd + thread_id);
  //     printPair(nowpair, n_seq);
  // }
  L_ID_TYPE *succ_ipair = new L_ID_TYPE[n_seq];
  L_ID_TYPE pos;
  for (int l = 1; l <= D_NUM_LABEL; l++) {
    flag = 1;
    pos = D_NUM_LABEL * thread_id + l - 1;
    for (int i = 0; i < n_seq; i++) {
      p = succT[get_id(i, l, nowpair.ipair[i], Tx, Ty, Tz)];
      if (p == D_INVALID_LID) {
        flag = 0;
        break;
      }
      succ_ipair[i] = p;
    }
    if (flag) {
      for (L_ID_TYPE i = 0; i < n_seq; i++) {
        new_pairs[pos].ipair[i] = succ_ipair[i];
      }
      new_pairs[pos].id = D_INVALID_PID;
      new_pairs[pos].level = nowpair.level + 1;
      new_pairs[pos].pred = nowpair.id;
      new_pairs[pos].state = PAIR_STATE::ACTIVE;
      valid_cnt++;
      // if(!thread_id) printPair(new_pairs[pos],n_seq);
    } else {
      new_pairs[pos].state = PAIR_STATE::INACTIVE;
    }
  }
  // if(!thread_id) printf("\nvalid: %ld\n",valid_cnt);
  delete succ_ipair;
}

__global__ void find_lcs(L_ID_TYPE len_seq, char *seqs, L_ID_TYPE len_lcs, char *lcss, Record *pairs, P_ID_TYPE ac_size,
                         P_ID_TYPE ac_hd) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= ac_size) return;

  P_ID_TYPE pid = ac_hd + thread_id;
  while (pid != D_INVALID_PID) {
    // if(!thread_id) printf("%ld\n", pid);
    Record &nowp = pairs[pid];
    // printf("pred: %ld, level: %ld letter: %c\n", nowp.pred, nowp.level, seqs[nowp.ipair[0]]);
    lcss[get_str_hd(thread_id, len_lcs + 2) + nowp.level] = seqs[nowp.ipair[0]];
    pid = nowp.pred;
  }
  lcss[get_str_hd(thread_id, len_lcs + 2) + len_lcs + 1] = '\0';
}

void FAST_LCS::calc_LCS() {
  init();

  auto _outt =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] bg fastlcs_cu " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001)
            << "\n";
  // Step 1
  build_succT();
  // printT();

  // Step 2
  get_init_ipair();

  P_ID_TYPE max_size;
  for (int O = 2;; O++) {
    // Step 3.1
    max_size = active_size * NUM_LABEL;
    // printP(a_pairs, pair_cnt);
    // printf("%ld %ld %ld %ld %ld\n", pair_cnt, active_hd, active_tl, active_size, max_size);

    find_dir_pair<<<thread_block_size((int)active_size, BLOCK_THREAD_DIM), BLOCK_THREAD_DIM>>>(
        n_seq, d_succT, m_level_pairs, m_pairs, active_size, active_hd, active_tl, Tx, Ty, Tz);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    CHK_CU(cudaDeviceSynchronize());

    // Step 3.2 pruning operations
    prune<<<thread_block_size((int)max_size, BLOCK_THREAD_DIM), BLOCK_THREAD_DIM>>>(n_seq, max_size, m_flag,
                                                                                    m_level_pairs);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    CHK_CU(cudaDeviceSynchronize());
    P_ID_TYPE new_active_hd = active_tl;
    bool has_new = false;
    for (P_ID_TYPE i = 0; i < max_size; i++) {
      if (!a_flag[i]) continue;
      has_new = true;
      a_level_pairs[i].id = pair_cnt;
      a_level_pairs[i].state = PAIR_STATE::INACTIVE;
      a_pairs[active_tl++] = a_level_pairs[i];
      pair_cnt++;
    }
    if (has_new) {
      active_hd = new_active_hd;
      active_size = active_tl - active_hd;
    } else {
      break;
    }
  }

  // Step 4. Compute r = the maximum level
  lcs_len = a_pairs[pair_cnt - 1].level;
  lcs_cnt = active_size;
  CHK_CU(cudaMalloc(&d_lcss, (lcs_len + 2) * active_size * sizeof(char)));
  find_lcs<<<thread_block_size((unsigned long)active_size, 256UL), 256UL>>>(max_len, d_seqs, lcs_len, d_lcss, m_pairs,
                                                                            active_size, active_hd);
  CHK_CU(cudaDeviceSynchronize());

  h_lcss = new char[(lcs_len + 2) * active_size];
  CHK_CU(cudaMemcpy(h_lcss, d_lcss, (lcs_len + 2) * active_size * sizeof(char), cudaMemcpyDeviceToHost));
  // printf("Step 4 finished.\n");
  // printf("LCS length: %ld, count: %ld\n", lcs_len, lcs_cnt);
  _outt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cout << "[PF] ed fastlcs_cu " << std::fixed << std::setprecision(6) << (double)(_outt.count() * 0.000001)
            << "\n";
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
  char s[MAX_LENGTH + 2];
  vector<char *> seqs;
  scanf("%d", &n);
  FAST_LCS lcs(n);
  getchar();
  s[0] = 'x';
  for (int i = 0; i < lcs.n_seq; i++) {
    fgets(s + 1, MAX_LENGTH, stdin);
    if (s[strlen(s) - 1] == '\n') {
      s[strlen(s) - 1] = '\0';
    }
    seqs.push_back(new char[strlen(s) + 2]);
    strcpy(seqs[seqs.size() - 1], s);
    lcs.lens[i] = strlen(s + 1);
    lcs.max_len = max(lcs.max_len, lcs.lens[i]);
  }

  lcs.h_seqs = new char[(lcs.max_len + 2) * lcs.n_seq];
  for (int i = 0; i < lcs.n_seq; i++) {
    strcpy(lcs.h_seqs + get_str_hd(i, lcs.max_len + 2), seqs[i]);
  }

  // for (int i = 0; i < lcs.n_seq; i++) {
  //   printf("%ld %s\n", lcs.lens[i], lcs.h_seqs + get_str_hd(i, lcs.max_len + 2) + 1);
  // }

  lcs.calc_LCS();
  printf("%ld\n%ld\n", lcs.lcs_len, lcs.lcs_cnt);
  // lcs.printLCS();

  return 0;
}
